"""Install NVIDIA's checksum-verified CUDA 13.3 redistributables in a CI temp dir."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
COMPONENTS = ["cuda_nvcc", "cuda_crt", "cuda_cudart", "libnvvm", "cccl",
              "libcublas", "cuda_cuobjdump", "cuda_nvdisasm", "cuda_nvml_dev", "cuda_profiler_api"]


def install(output: Path):
    with urllib.request.urlopen(BASE + "redistrib_13.3.0.json", timeout=60) as response:
        manifest = json.load(response)
    output.mkdir(parents=True, exist_ok=False)
    downloads = output / "downloads"
    downloads.mkdir()
    records = []
    for name in COMPONENTS:
        component = manifest[name]
        item = component.get("windows-x86_64") or component.get("windows-all")
        if item is None:
            raise RuntimeError(f"No Windows redistributable for {name}")
        archive = downloads / Path(item["relative_path"]).name
        print(f"Downloading {name} {component['version']}", flush=True)
        with urllib.request.urlopen(BASE + item["relative_path"], timeout=120) as response, archive.open("wb") as target:
            shutil.copyfileobj(response, target)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError(f"SHA256 mismatch: {name}")
        # NVIDIA archives contain exactly one component-root directory.
        license_file = output / "licenses" / "nvidia" / (name + ".txt")
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                parts = Path(entry.filename).parts
                if len(parts) < 2 or entry.is_dir():
                    continue
                relative = Path(*parts[1:])
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError("Invalid archive path")
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(entry) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
                # Every archive calls its root notice LICENSE; keep them separately.
                if relative == Path("LICENSE"):
                    license_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(target, license_file)
        if not license_file.is_file():
            raise RuntimeError(f"Missing NVIDIA redistribution notice: {name}")
        records.append({"component": name, "version": component["version"], **item})
    (output / "installed-components.json").write_text(json.dumps(records, indent=2))
    for required in ["bin/nvcc.exe", "bin/cuobjdump.exe", "bin/nvdisasm.exe", "include/cuda.h", "lib/x64/cuda.lib", "lib/x64/cudart_static.lib"]:
        if not (output / required).is_file():
            raise RuntimeError(f"Incomplete CUDA toolkit: {required}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    install(parser.parse_args().output.resolve())
