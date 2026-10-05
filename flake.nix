{
  description = "Dev shell for Qwen3.8-27B drafter + CUDA kernel optimization work (RTX 3090, sm_86)";

  # Same nixpkgs revision as the deployment flake, so the CUDA closure is shared.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/624af665418d3c65d544145b4d34ad696439570e";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
          cudaCapabilities = [ "8.6" ];
          cudaEnableForwardCompat = false;
        };
      };
      cp = pkgs.cudaPackages;
      cudaJoined = pkgs.symlinkJoin {
        name = "cuda-joined";
        paths = pkgs.lib.concatMap (p: [ (pkgs.lib.getBin p) (pkgs.lib.getDev p) (pkgs.lib.getLib p) ] ++ pkgs.lib.optional (p ? include) p.include ++ pkgs.lib.optional (p ? static) p.static) (with cp; [
          cuda_nvcc
          cuda_cudart
          cccl
          libcublas
          cuda_nvml_dev
          cuda_profiler_api
          cuda_cupti
          cuda_nvrtc
          cuda_cuobjdump
          cuda_nvdisasm
          cuda_sanitizer_api
        ]);
      };
    in
    {
      devShells.${system}.default = (pkgs.mkShell.override { stdenv = cp.backendStdenv; }) {
        packages = [
          cudaJoined
          cp.nsight_systems
          cp.nsight_compute
          pkgs.cmake
          pkgs.ninja
          pkgs.ccache
          pkgs.pkg-config
          pkgs.openssl
        ];
        CUDA_HOME = "${cudaJoined}";
        CUDAToolkit_ROOT = "${cudaJoined}";
        shellHook = ''
          # host NVIDIA user-space driver libraries in <repo>/driver-libs (also found from llama.cpp/)
          root=$PWD
          while [ "$root" != / ] && [ ! -e "$root/docs/DEPLOYMENT.md" ]; do root=$(dirname "$root"); done
          export LD_LIBRARY_PATH=$root/driver-libs''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        '';
      };
    };
}
