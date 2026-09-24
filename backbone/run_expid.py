import argparse
import gc
import logging
import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from datetime import datetime

FUXICTR_ROOT = Path(__file__).resolve().parents[1]
os.chdir(FUXICTR_ROOT)
sys.path.insert(0, str(FUXICTR_ROOT))
sys.path.insert(0, str(FUXICTR_ROOT / "method"))

from backbone import fuxictr_version  # noqa: E402,F401
import torch  # noqa: E402
from torch.utils.data import DataLoader as TorchDataLoader  # noqa: E402
from torch.utils.data.dataloader import default_collate  # noqa: E402
from fuxictr.features import FeatureMap  # noqa: E402
from fuxictr.pytorch.dataloaders import RankDataLoader  # noqa: E402
from fuxictr.pytorch.dataloaders.parquet_dataloader import (  # noqa: E402
    ParquetDataset,
)
from fuxictr.pytorch.torch_utils import seed_everything  # noqa: E402
from fuxictr.utils import (  # noqa: E402
    load_dataset_config,
    load_model_config,
    print_to_json,
    print_to_list,
    set_logger,
)


def resolve_model_class(model_name):
    """Resolve a frozen native FuxiCTR backbone from this benchmark."""

    if not str(model_name).isidentifier():
        raise ValueError(f"Invalid model name: {model_name!r}")
    model_path = FUXICTR_ROOT / "backbone" / model_name / f"{model_name}.py"
    if not model_path.is_file():
        raise AttributeError(
            f"Model {model_name!r} is not among the bundled backbones at {model_path}"
        )
    module_name = f"_numemb_native_{model_name}"
    spec = spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load native model from {model_path}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    logging.info("Loaded unmodified native model: %s", model_path)
    return getattr(module, model_name)


class PackedPinnedParquetDataLoader(TorchDataLoader):
    """Non-streaming Parquet loader with one pinned 2-D tensor per batch.

    It preserves the exact values and row order of FuxiCTR's standard Parquet
    loader.  Column views are materialized only after a single asynchronous
    host-to-device copy in ``CUDAPrefetchLoader``.
    """

    def __init__(self, feature_map, data_path, batch_size=32, shuffle=False,
                 num_workers=1, **kwargs):
        if not data_path.endswith(".parquet"):
            data_path += ".parquet"
        self.feature_map = feature_map
        self.dataset = ParquetDataset(feature_map, data_path)
        loader_kwargs = dict(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=default_collate,
            pin_memory=True,
        )
        if num_workers > 0:
            loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
        super().__init__(**loader_kwargs)
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size

    def __len__(self):
        return self.num_batches


class CUDAPrefetchLoader:
    """Re-iterable double-buffered loader producing the standard input dict."""

    def __init__(self, loader, feature_map, device):
        self.loader = loader
        self.feature_map = feature_map
        self.device = torch.device(device)
        self.columns = list(feature_map.features.keys()) + feature_map.labels

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        iterator = iter(self.loader)
        stream = torch.cuda.Stream(device=self.device)

        def preload():
            try:
                host_batch = next(iterator)
            except StopIteration:
                return None
            with torch.cuda.stream(stream):
                device_batch = host_batch.to(self.device, non_blocking=True)
                return {
                    name: device_batch[:, self.feature_map.get_column_index(name)]
                    for name in self.columns
                }

        next_batch = preload()
        while next_batch is not None:
            torch.cuda.current_stream(self.device).wait_stream(stream)
            batch = next_batch
            for tensor in batch.values():
                tensor.record_stream(torch.cuda.current_stream(self.device))
            next_batch = preload()
            yield batch


def enable_exact_runtime(model, mode):
    """Enable a deployment runtime without changing model parameters/layers.

    This is deliberately opt-in. Dataset validation remains a separate data
    preflight responsibility; the model class keeps validation enabled by
    default outside this deployment path.
    """
    if mode == "none":
        return
    if mode not in (
        "packed-only",
        "module-compile-packed",
        "default-packed",
        "reduce-overhead",
        "reduce-overhead-packed",
        "reduce-overhead-packed-fused",
        "reduce-overhead-packed-tf32-fused",
        "reduce-overhead-packed-bf16-fused",
        "portable-packed",
    ):
        raise ValueError(f"unsupported compile runtime: {mode}")
    torch._dynamo.config.cache_size_limit = 64
    use_fused_adam = mode in (
        "reduce-overhead-packed-fused",
        "reduce-overhead-packed-tf32-fused",
        "reduce-overhead-packed-bf16-fused",
    )
    use_tf32 = mode == "reduce-overhead-packed-tf32-fused"
    use_bf16 = mode == "reduce-overhead-packed-bf16-fused"
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    scalar_count = 0
    scalar_module = None
    for module in model.modules():
        if module.__class__.__name__ == "FeatureEmbeddingDict":
            module.categorical_lookup_mode = "auto"
            module.reuse_context_embeddings = True
        elif module.__class__.__name__ == "FastCoupledEmbedding":
            scalar_count += 1
            scalar_module = module
            module.validate_inputs = False
            module.set_execution_mode("eager")
        elif module.__class__.__name__ == "DEEREmbedding":
            module.set_execution_mode("reference")
    if scalar_count != 1:
        raise RuntimeError(
            "reduce-overhead runtime is restricted to one ScalarLens module; "
            f"found {scalar_count}"
        )
    if mode == "packed-only":
        logging.info(
            "Enabled ScalarLens exact packed runtime without compiler: input "
            "checks disabled after dataset preflight; parameters and objective "
            "unchanged."
        )
        return
    if mode == "module-compile-packed" or (
        mode == "portable-packed" and model.__class__.__name__ == "DeepFM"
    ):
        wrapped = torch.compile(
            scalar_module, fullgraph=True, dynamic=False,
            mode="reduce-overhead",
        )
        for owner in model.modules():
            layers = getattr(owner, "numeric_embedding_layers", None)
            if layers is None:
                continue
            for key, value in layers.items():
                if value is scalar_module:
                    layers[key] = wrapped
                    logging.info(
                        "Enabled ScalarLens FP32 module-only compiled packed "
                        "runtime; full backbone remains eager."
                    )
                    return
        raise RuntimeError("ScalarLens owner slot not found")

    compile_mode = "reduce-overhead"
    if mode == "default-packed":
        compile_mode = None
    elif mode == "portable-packed":
        # The portable path uses whole-step reduce-overhead for DCNv2 and the
        # module-only branch above for DeepFM.  The latter avoids an invalid
        # accelerator Triton reduction for AutoML-A's short-field shape.
        compile_mode = "reduce-overhead"

    compile_kwargs = dict(fullgraph=False, dynamic=False)
    if compile_mode is not None:
        compile_kwargs["mode"] = compile_mode
    original_forward = model.forward
    if use_bf16:
        if model.__class__.__name__ != "DCNv2" or not hasattr(model, "parallel_dnn"):
            raise RuntimeError("BF16 endpoint screen currently supports parallel DCNv2 only")

        def cast_parallel_output(_module, _inputs, output):
            # DCNv2's cross branch remains FP32 on this accelerator while the
            # DNN branch is autocast to BF16.  Aligning only the branch output
            # makes concatenation legal without changing parameters or layers.
            return output.float()

        model.parallel_dnn.register_forward_hook(cast_parallel_output)

        def precision_forward(*args, **kwargs):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = original_forward(*args, **kwargs)
            if isinstance(output, dict) and "y_pred" in output:
                output = dict(output)
                output["y_pred"] = output["y_pred"].float()
            return output

        forward_for_compile = precision_forward
    else:
        forward_for_compile = original_forward
    model.forward = torch.compile(forward_for_compile, **compile_kwargs)

    if use_fused_adam:
        if not isinstance(model.optimizer, torch.optim.Adam):
            raise TypeError("fused runtime currently requires torch.optim.Adam")
        defaults = model.optimizer.defaults
        model.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=defaults["lr"],
            betas=defaults["betas"],
            eps=defaults["eps"],
            weight_decay=defaults["weight_decay"],
            amsgrad=defaults.get("amsgrad", False),
            maximize=defaults.get("maximize", False),
            fused=True,
        )

    def compiled_train_step(batch_data):
        model.optimizer.zero_grad(set_to_none=True)
        return_dict = model(batch_data)
        y_true = model.get_labels(batch_data)
        loss = model.compute_loss(return_dict, y_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), model._max_gradient_norm)
        model.optimizer.step()
        return loss

    model.train_step = torch.compile(compiled_train_step, **compile_kwargs)
    precision = "BF16" if use_bf16 else ("TF32" if use_tf32 else "FP32")
    logging.info(
        "Enabled ScalarLens %s compiled runtime (mode=%s): input checks "
        "disabled after dataset preflight; parameters and objective unchanged.",
        precision,
        (compile_mode or "default")
        + ("+fused-adam" if use_fused_adam else "")
        + ("+tf32" if use_tf32 else "")
        + ("+bf16-autocast" if use_bf16 else ""),
    )


def make_data_iterators(feature_map, params, runtime_mode, stage, device):
    """Build standard or packed/prefetched iterators without changing samples."""
    loader_params = dict(params)
    use_packed = runtime_mode in (
        "packed-only",
        "module-compile-packed",
        "default-packed",
        "reduce-overhead-packed",
        "reduce-overhead-packed-fused",
        "reduce-overhead-packed-tf32-fused",
        "reduce-overhead-packed-bf16-fused",
        "portable-packed",
    )
    if use_packed:
        if params.get("data_format") != "parquet" or params.get("streaming"):
            raise ValueError(
                "reduce-overhead-packed requires non-streaming Parquet data"
            )
        loader_params["data_loader"] = PackedPinnedParquetDataLoader
    iterator = RankDataLoader(
        feature_map, stage=stage, **loader_params
    ).make_iterator()
    if not use_packed:
        return iterator
    if stage == "train":
        train_gen, valid_gen = iterator
        return (
            CUDAPrefetchLoader(train_gen, feature_map, device),
            CUDAPrefetchLoader(valid_gen, feature_map, device),
        )
    return CUDAPrefetchLoader(iterator, feature_map, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/")
    parser.add_argument("--expid", default="NumEmbDNN_field")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="Optional dataset override, e.g. AvazuNumSmoke for a smoke run.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--model-root",
        default=None,
        help="Optional checkpoint root override for isolated benchmark runs.",
    )
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--compile-runtime",
        choices=(
            "none",
            "packed-only",
            "module-compile-packed",
            "default-packed",
            "reduce-overhead",
            "reduce-overhead-packed",
            "reduce-overhead-packed-fused",
            "reduce-overhead-packed-tf32-fused",
            "reduce-overhead-packed-bf16-fused",
            "portable-packed",
        ),
        default="none",
        help="Opt-in ScalarLens deployment runtime; use only after data preflight.",
    )
    parser.add_argument(
        "--results-file",
        default="results.csv",
        help="Result file path; useful for isolated outputs in parallel runs.",
    )
    args = parser.parse_args()

    params = load_model_config(args.config, args.expid)
    if args.dataset_id:
        params["dataset_id"] = args.dataset_id
    params.update(load_dataset_config(args.config, params["dataset_id"]))
    params["gpu"] = args.gpu
    if args.seed is not None:
        params["seed"] = args.seed
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.model_root is not None:
        params["model_root"] = args.model_root
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params["seed"])

    data_dir = os.path.join(params["data_root"], params["dataset_id"])
    feature_map = FeatureMap(params["dataset_id"], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"), params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    model_class = resolve_model_class(params["model"])
    model = model_class(feature_map, **params)
    enable_exact_runtime(model, args.compile_runtime)
    model.count_parameters()
    train_gen, valid_gen = make_data_iterators(
        feature_map, params, args.compile_runtime, stage="train",
        device=model.device,
    )
    model.fit(train_gen, validation_data=valid_gen, **params)
    valid_result = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()

    test_result = {}
    if params.get("test_data") and not args.skip_test:
        test_gen = make_data_iterators(
            feature_map, params, args.compile_runtime, stage="test",
            device=model.device,
        )
        test_result = model.evaluate(test_gen)

    result_filename = Path(args.results_file)
    result_filename.parent.mkdir(parents=True, exist_ok=True)
    with result_filename.open("a+") as file:
        file.write(
            " {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n".format(
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                " ".join(sys.argv),
                args.expid,
                params["dataset_id"],
                "N.A.",
                print_to_list(valid_result),
                print_to_list(test_result),
            )
        )


if __name__ == "__main__":
    main()
