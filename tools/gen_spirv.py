# 由 GLSL 着色器再生 SPIR-V 头(修改 shaders/*.comp 后运行):
#   third_party/glslang/bin/glslang.exe -V --target-env vulkan1.1 \
#       src/backends/vulkan/shaders/X.comp -o X.spv && python tools/gen_spirv.py
import io, sys, os

def emit(name, spv_path, out_dir):
    with open(spv_path, 'rb') as f:
        data = f.read()
    words = [int.from_bytes(data[i:i+4], 'little') for i in range(0, len(data), 4)]
    NL = chr(10)
    out = []
    out.append('// 由 shaders/%s.comp 经 glslang 生成(SPIR-V, vulkan1.1);' % name)
    out.append('// 修改着色器后: 见 tools/gen_spirv.py 头部注释再生')
    out.append('#pragma once')
    out.append('#include <cstdint>')
    out.append('#include <cstddef>')
    out.append('namespace znn { namespace vulkan_backend { namespace spirv {')
    out.append('inline const uint32_t %s[] = {' % name)
    for i in range(0, len(words), 8):
        out.append('    ' + ' '.join('0x%08xu,' % w for w in words[i:i+8]))
    out.append('};')
    out.append('inline const size_t %s_bytes = sizeof(%s);' % (name, name))
    out.append('}}}') 
    with open(os.path.join(out_dir, '%s.h' % name), 'w', encoding='utf-8', newline='') as f:
        f.write(NL.join(out) + NL)
    print('gen', name, len(words), 'words')

if __name__ == '__main__':
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'src/backends/vulkan/gen'
    import glob, tempfile, subprocess
    for comp in glob.glob('src/backends/vulkan/shaders/*.comp'):
        name = os.path.splitext(os.path.basename(comp))[0]
        spv = os.path.join(tempfile.gettempdir(), name + '.spv')
        r = subprocess.run(['third_party/glslang/bin/glslang.exe', '-V',
                            '--target-env', 'vulkan1.1', comp, '-o', spv],
                           capture_output=True)
        if r.returncode != 0:
            print('FAILED', name, r.stderr.decode()[:200]); sys.exit(1)
        emit(name, spv, out_dir)
