#!/usr/bin/env bash
# ============================================================================
# restore_deps.sh — 一键重建 gitignore 的外部依赖(third_party/ 与 data/)
#
# 幂等: 已就位的部分自动跳过。用法:
#   bash tools/restore_deps.sh          # 全部(Eigen/oneDNN/Vulkan/glslang/MNIST)
#   bash tools/restore_deps.sh eigen   # 只装单项: eigen|onednn|vulkan|glslang|mnist
#
# 各项来源:
#   eigen    gitlab libeigen 3.4.0 头文件(~3MB)
#   onednn   优先 TEMP 残留, 兜底 GitHub v3.13.4 源码 tar 提取头 +
#            生成头取 tools/onednn-generated/(随仓库走, 免 cmake 配置)
#   vulkan   优先 TEMP 残留, 兜底 GitHub KhronosGroup v1.4.304(含 vk_video)
#   glslang  优先 TEMP 残留, 兜底 GitHub releases 16.6.0 win-x64
#   mnist    ossci + googleapis 逐文件 → deepai 整包兜底(精确字节数校验)
# ============================================================================
set -u
cd "$(dirname "$0")/.."   # 仓库根

TMP="${TMPDIR:-/tmp}"
winpath() { cygpath -m "$1" 2>/dev/null || echo "$1"; }  # 给 python 等原生程序的路径
fsize() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0; }  # GNU/BSD 通用
PY="$(command -v python || command -v python3 || echo python3)"  # mac 无 python 只有 python3
fetch() {  # fetch <url> <dest> [最小字节]
    local url="$1" dest="$2" min="${3:-1000}"
    for i in 1 2 3; do
        echo "  [$i/3] $url"
        curl -sL --max-time 300 -o "$dest" "$url" || true
        if [ -f "$dest" ] && [ "$(fsize "$dest")" -ge "$min" ]; then
            return 0
        fi
        sleep 3
    done
    echo "FAILED: $url" >&2
    return 1
}

# ---- Eigen 3.4.0(头文件, ~3MB) ----
do_eigen() {
    if [ -f third_party/eigen/Eigen/Dense ]; then echo "[eigen] 已就位"; return 0; fi
    echo "[eigen] 下载 3.4.0..."
    fetch "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz" \
          "$TMP/eigen.tar.gz" 1000000 || return 1
    tar xzf "$TMP/eigen.tar.gz" -C "$TMP"
    mkdir -p third_party/eigen
    cp -r "$TMP/eigen-3.4.0/Eigen" third_party/eigen/
    cp -r "$TMP/eigen-3.4.0/signatures" third_party/eigen/ 2>/dev/null || true
    [ -f third_party/eigen/Eigen/Dense ] && echo "[eigen] OK" || { echo "[eigen] FAIL"; return 1; }
}

# ---- oneDNN 头(vendored 类型 + 构建生成头) ----
do_onednn() {
    if [ -f third_party/onednn/include/dnnl.h ]; then echo "[onednn] 已就位"; return 0; fi
    mkdir -p third_party/onednn/include
    # 生成头(dnnl_config/version/hash): 随仓库走, 永不依赖 TEMP
    cp tools/onednn-generated/*.h third_party/onednn/include/ 2>/dev/null || true
    # 类型头: 优先 TEMP 残留, 兜底 GitHub 源码
    if [ -d "$TMP/oneDNN-3.13.4/include/oneapi" ]; then
        echo "[onednn] 从 TEMP 恢复..."
        cp -r "$TMP/oneDNN-3.13.4/include/"* third_party/onednn/include/
    else
        echo "[onednn] 下载 v3.13.4..."
        fetch "https://github.com/uxlfoundation/oneDNN/archive/refs/tags/v3.13.4.tar.gz" \
              "$TMP/onednn.tar.gz" 2000000 || return 1
        tar xzf "$TMP/onednn.tar.gz" -C "$TMP"
        cp -r "$TMP/oneDNN-3.13.4/include/"* third_party/onednn/include/
    fi
    # 生成头覆盖(TEMP 的 odnn-build 版本更新则用它)
    if [ -d "$TMP/odnn-build/include/oneapi/dnnl" ]; then
        cp "$TMP/odnn-build/include/oneapi/dnnl/"*.h \
           third_party/onednn/include/oneapi/dnnl/ 2>/dev/null || true
    fi
    [ -f third_party/onednn/include/dnnl.h ] \
      && [ -f third_party/onednn/include/oneapi/dnnl/dnnl_config.h ] \
      && echo "[onednn] OK" || { echo "[onednn] FAIL"; return 1; }
}

# ---- Vulkan 头(含 vk_video) ----
do_vulkan() {
    if [ -f third_party/vulkan/include/vulkan/vulkan.h ]; then echo "[vulkan] 已就位"; return 0; fi
    mkdir -p third_party/vulkan/include
    if [ -d "$TMP/Vulkan-Headers-1.4.304/include/vulkan" ]; then
        echo "[vulkan] 从 TEMP 恢复..."
        cp -r "$TMP/Vulkan-Headers-1.4.304/include/"* third_party/vulkan/include/
    else
        echo "[vulkan] 下载 v1.4.304..."
        fetch "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.4.304.tar.gz" \
              "$TMP/vh.tar.gz" 500000 || return 1
        tar xzf "$TMP/vh.tar.gz" -C "$TMP"
        cp -r "$TMP/Vulkan-Headers-1.4.304/include/"* third_party/vulkan/include/
    fi
    [ -f third_party/vulkan/include/vulkan/vulkan.h ] \
      && [ -d third_party/vulkan/include/vk_video ] \
      && echo "[vulkan] OK" || { echo "[vulkan] FAIL"; return 1; }
}

# ---- glslang(SPIR-V 编译器, 仅 Windows 构建需要) ----
do_glslang() {
    case "$(uname -s)" in Darwin|Linux) echo "[glslang] 跳过(SPIR-V 已随仓库, 仅 Windows 重生成时需要)"; return 0 ;; esac
    if [ -f third_party/glslang/bin/glslang.exe ]; then echo "[glslang] 已就位"; return 0; fi
    if [ -f "$TMP/glslang-bin/bin/glslang.exe" ]; then
        echo "[glslang] 从 TEMP 恢复..."
        mkdir -p third_party/glslang/bin
        cp "$TMP/glslang-bin/bin/glslang.exe" third_party/glslang/bin/
    else
        echo "[glslang] 下载 16.6.0 win-x64..."
        fetch "https://github.com/KhronosGroup/glslang/releases/download/16.6.0/glslang-16.6.0-windows-x86_64-release.zip" \
              "$TMP/glslang.zip" 5000000 || return 1
        local tmpw; tmpw=$(winpath "$TMP")
        $PY -c "import zipfile; zipfile.ZipFile('$tmpw/glslang.zip').extractall('$tmpw/glslang-bin')" 2>/dev/null \
        || unzip -o "$TMP/glslang.zip" -d "$TMP/glslang-bin" >/dev/null
        mkdir -p third_party/glslang/bin
        find "$TMP/glslang-bin" -name glslang.exe -exec cp {} third_party/glslang/bin/ \;
    fi
    [ -f third_party/glslang/bin/glslang.exe ] && echo "[glslang] OK" || { echo "[glslang] FAIL"; return 1; }
}

# ---- MNIST 数据集(4 个 idx 文件; 逐文件双镜像 → deepai 整包兜底, 精确字节数
#      校验防 404 XML 假成功: 直链镜像会返回 ~300 字节 S3 NoSuchKey 错误页) ----
do_mnist() {
    mkdir -p data/mnist
    local files="train-images-idx3-ubyte:47040016 train-labels-idx1-ubyte:60008 \
t10k-images-idx3-ubyte:7840016 t10k-labels-idx1-ubyte:10008"
    local ok=1 missing=""
    for f in $files; do
        local name="${f%%:*}" want="${f##*:}"
        local sz; sz=$(fsize "data/mnist/$name")
        if [ "$sz" = "$want" ]; then continue; fi
        local got=0
        for base in "https://ossci-datasets.s3.amazonaws.com/mnist" \
                    "https://storage.googleapis.com/cvdf-datasets/mnist"; do
            rm -f "data/mnist/$name"
            if fetch "$base/$name" "data/mnist/$name" "$want"; then
                sz=$(fsize "data/mnist/$name")
                if [ "$sz" = "$want" ]; then got=1; break; fi
                echo "[mnist] $name 得 $sz 字节 != $want(404/截断), 换镜像"
            fi
        done
        [ "$got" = 1 ] || missing="$missing $name"
    done
    if [ -n "$missing" ]; then   # 兜底: deepai 整包(内为 .gz, 需解两层)
        local tmpw; tmpw=$(winpath "$TMP")
        echo "[mnist] 直链镜像失败($missing), 改用 deepai 整包..."
        fetch "https://data.deepai.org/mnist.zip" "$TMP/mnist.zip" 10000000 \
          || { echo "[mnist] FAIL"; return 1; }
        rm -rf "$TMP/mnist-ex"; mkdir -p "$TMP/mnist-ex"
        $PY -c "
import gzip, shutil, zipfile
z = zipfile.ZipFile('$tmpw/mnist.zip')
for n in z.namelist():
    with z.open(n) as s, open('$tmpw/mnist-ex/' + n[:-3], 'wb') as d:
        shutil.copyfileobj(gzip.open(s), d)
" || { echo "[mnist] 解包失败"; return 1; }
        for f in $files; do
            local name="${f%%:*}" want="${f##*:}"
            local sz; sz=$(fsize "$TMP/mnist-ex/$name")
            if [ "$sz" = "$want" ]; then mv "$TMP/mnist-ex/$name" "data/mnist/$name"
            else echo "[mnist] $name 整包内容异常($sz 字节)"; ok=0; fi
        done
    fi
    [ "$ok" = 1 ] && echo "[mnist] OK" || { echo "[mnist] FAIL"; return 1; }
}

# ---- 入口 ----
targets="${1:-all}"
if [ "$targets" = "all" ]; then
    targets="eigen onednn vulkan glslang mnist"
fi
rc=0
for t in $targets; do
    case "$t" in
        eigen)   do_eigen   || rc=1 ;;
        onednn)  do_onednn  || rc=1 ;;
        vulkan)  do_vulkan  || rc=1 ;;
        glslang) do_glslang || rc=1 ;;
        mnist)   do_mnist   || rc=1 ;;
        *) echo "未知项: $t" >&2; rc=1 ;;
    esac
done
if [ "$rc" = 0 ]; then
    echo "=== 依赖重建完成 ==="
else
    echo "=== 有项目失败(见上) ===" >&2
fi
exit $rc
