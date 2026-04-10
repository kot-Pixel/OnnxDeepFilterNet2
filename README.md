# ONNX Runtime

## QNN Rumtime

Onnx runtime 来去使用 CPU 进行推理的时候发现 CPU 占用的太高。

所以想尝试使用QNN 来使用 HTP 后端来进行加速。下面是将 DeepFliterNet2 三个onnx 模型转换为QNN Runtime

去执行。

```bash

setenvvar ONNX_MODEL_PATH "${QNN_SDK_ROOT}/examples/Models/erb_dec.onnx"

python3 -c "
import onnx
m = onnx.load('erb_dec.onnx')
for i in m.graph.input:
    if i.name.startswith('val_') and i.name in str(i): pass
    t = i.type.tensor_type
    dims = [d.dim_value if d.dim_value else d.dim_param for d in t.shape.dim]
    print(i.name, dims)
"

${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-onnx-converter   --input_network "${ONNX_MODEL_PATH}"   -n   -d "emb" "1,1,512"   -d "e3" "1,64,1,8" -d "e2" "1,64,1,8"   -d "e1" "1,64,1,16"   -d "e0" "1,64,1,32"   -d "buf_convt3" "1,64,1,1" -d "buf_c0out" "1,64,1,1"   -d "h_erb" "2,1,256"   -o "${ONNX_MODEL_PATH%.*}_qnn_model.cpp"


运行结果为: 2026-04-08 14:26:53,421 - 278 - INFO - Conversion complete!

-rwxrwxrwx 1 wdf wdf 3317760 Apr  8 14:26 erb_dec_qnn_model.bin
-rwxrwxrwx 1 wdf wdf  402529 Apr  8 14:26 erb_dec_qnn_model.cpp


上面为erb_dec.onnx

setenvvar ONNX_MODEL_PATH "${QNN_SDK_ROOT}/examples/Models/df_dec.onnx"

python3 -c "
import onnx                                                                                                 m = onnx.load('df_dec.onnx')
for i in m.graph.input:
    if i.name.startswith('val_') and i.name in str(i): pass
    t = i.type.tensor_type
    dims = [d.dim_value if d.dim_value else d.dim_param for d in t.shape.dim]
    print(i.name, dims)
"

${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-onnx-converter \
  --input_network "${ONNX_MODEL_PATH}" \
  -n \
  -d "emb" "1,1,512" \
  -d "c0" "1,64,1,96" \
  -d "buf_dfcp" "1,64,4,96" \
  -d "h_df" "2,1,256" \
  -o "${ONNX_MODEL_PATH%.*}_qnn_model.cpp"

-rwxrwxrwx 1 wdf wdf 3348480 Apr  8 14:30 df_dec_qnn_model.bin
-rwxrwxrwx 1 wdf wdf  309959 Apr  8 14:30 df_dec_qnn_model.cpp

上面为df_dec.onnx


setenvvar ONNX_MODEL_PATH "${QNN_SDK_ROOT}/examples/Models/enc.onnx"

python3 -c "
import onnx
m = onnx.load('enc.onnx')
for i in m.graph.input:
    if i.name.startswith('val_') and i.name in str(i): pass
    t = i.type.tensor_type
    dims = [d.dim_value if d.dim_value else d.dim_param for d in t.shape.dim]
    print(i.name, dims)
"

${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-onnx-converter \
  --input_network "${ONNX_MODEL_PATH}" \
  -n \
  -d "feat_erb" "1,1,1,32" \
  -d "feat_spec" "1,2,1,96" \
  -d "buf_erb0" "1,1,2,32" \
  -d "buf_erb1" "1,64,1,1" \
  -d "buf_erb2" "1,64,1,1" \
  -d "buf_erb3" "1,64,1,1" \
  -d "buf_df0" "1,2,2,96" \
  -d "buf_df1" "1,64,1,1" \
  -d "h_enc" "1,1,256" \
  -o "${ONNX_MODEL_PATH%.*}_qnn_model.cpp"

-rwxrwxrwx 1 wdf wdf 1966080 Apr  8 14:34 enc_qnn_model.bin
-rwxrwxrwx 1 wdf wdf  333441 Apr  8 14:34 enc_qnn_model.cpp


上面为 enc.onnx 转化为的 qnn runtime 转换之后time.

```

## 非 Stream onnx 模型转为 QNN

```bash

 cp /mnt/d/DeepFilterNet/DeepFilterNet/df/scripts/replace_einsum_onnx.py .

python3 replace_einsum_onnx.py erb_dec.onnx erb_dec_no_einsum.onnx

setenvvar ONNX_MODEL_PATH "${QNN_SDK_ROOT}/examples/Models/erb_dec_no_einsum.onnx

export N=128

${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-onnx-converter   --input_network "${ONNX_MODEL_PATH}"   -n   -s S "${N}"   -d "emb" "1,${N},512"   -d "e3" "1,64,${N},8"   -d "e2" "1,64,${N},8"   -d "e1" "1,64,${N},16"   -d "e0" "1,64,${N},32"   -o "${ONNX_MODEL_PATH%.*}_qnn_model.cpp"

python3 "${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator"   -c "$(pwd)/erb_dec_no_einsum_qnn_model.cpp"   -b "$(pwd)/erb_dec_no_einsum_qnn_model.bin
"   -o "$(pwd)/model_libs"   -t aarch64-android

-----------------------------------------------------------------------

python3 replace_einsum_onnx.py enc.onnx enc_no_einsum.onnx

setenvvar ONNX_MODEL_PATH "${QNN_SDK_ROOT}/examples/Models/enc_no_einsum.onnx"

export N=128

${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-onnx-converter   --input_network "${ONNX_MODEL_PATH}"   -n   -s S "${N}"   -d "feat_erb" "1,1,${N},32"   -d "feat_spec" "1,2,${N},96"   -o "${ONNX_MODEL_PATH%.*}_qnn_model.cpp"


python3 "${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator"   -c "$(pwd)/enc_no_einsum_qnn_model.cpp"   -b "$(pwd)/enc_no_einsum_qnn_model.bin"   -o "
$(pwd)/model_libs"   -t aarch64-android

-----------------------------------------------------------------------

python3 replace_einsum_onnx.py df_dec.onnx df_dec_no_einsum.onnx

setenvvar ONNX_MODEL_PATH "${QNN_SDK_ROOT}/examples/Models/df_dec_no_einsum.onnx"

export N=128

${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-onnx-converter   --input_network "${ONNX_MODEL_PATH}"   -n   -s S "${N}"   -d "emb" "1,${N},512"   -d "c0" "1,64,${N},96"   -o "${ONNX_MODEL_PATH%.*}_qnn_model.cpp"


python3 "${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator"   -c "$(pwd)/df_dec_no_einsum_qnn_model.cpp"   -b "$(pwd)/df_dec_no_einsum_qnn_model.bin"   -o "$(pwd)/model_libs"   -t aarch64-android


生成的QNN的so库的为：

qnn@localhost:~/qairt/2.45.0.260326/examples/Models/model_libs/aarch64-android$ ls -alh
total 62M
drwxrwxr-x 2 qnn qnn 4.0K Apr  8 16:25 .
drwxrwxr-x 3 qnn qnn 4.0K Apr  8 15:57 ..
-rwxr-xr-x 1 qnn qnn  31M Apr  8 15:57 libdf_dec_no_einsum_qnn_model.so
-rwxr-xr-x 1 qnn qnn  11M Apr  8 16:18 libenc_no_einsum_qnn_model.so
-rwxr-xr-x 1 qnn qnn  21M Apr  8 16:25 liberb_dec_no_einsum_qnn_model.so

```

现在对应

## 生成input_list.txt的方法

```bash
python3 -c "
import onnx
m = onnx.load('erb_dec.onnx')
for i in m.graph.input:
    if i.name.startswith('val_') and i.name in str(i): pass
    t = i.type.tensor_type
    dims = [d.dim_value if d.dim_value else d.dim_param for d in t.shape.dim]
    print(i.name, dims)
"
```

查看graph的shape...

erb_dec.onnx的输入是这些。

```bash
emb [1, 'S', 512]
e3 [1, 64, 'S', 8]
e2 [1, 64, 'S', 8]
e1 [1, 64, 'S', 16]
e0 [1, 64, 'S', 32]
```

这里假设这个S = 1, 执行下面的脚本。之后这里就会出现erb_inputs

```bash
import numpy as np
import os
out_dir = "erb_inputs"
os.makedirs(out_dir, exist_ok=True)
shapes = [
    ("emb", (1, 1, 512)),
    ("e3",  (1, 64, 1, 8)),
    ("e2",  (1, 64, 1, 8)),
    ("e1",  (1, 64, 1, 16)),
    ("e0",  (1, 64, 1, 32)),
]
for name, sh in shapes:
    np.random.randn(*sh).astype(np.float32).tofile(os.path.join(out_dir, f"{name}.raw"))
```

这里将下面的内容写入到input_list.txt中去。

```bash
erb_inputs/emb.raw erb_inputs/e3.raw erb_inputs/e2.raw erb_inputs/e1.raw erb_inputs/e0.raw
```

使用HTP作为后端的时候，还需要将link 对应dsp的库。

```bash
~/qairt/2.45.0.260326/lib/hexagon-v68/unsigned$ cp ./libSnpeHtpV68Skel.so  /mnt/d/Qnn

export VENDOR_LIB=/vendor/lib64/

export LD_LIBRARY_PATH=/data/local/tmp/qnn:/vendor/dsp/cdsp:$VENDOR_LIB
export ADSP_LIBRARY_PATH="/data/local/tmp/qnn;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"

./qnn-sample-app  --backend ./libQnnHtp.so  --model ./liberb_dec_no_einsum_qnn_model.so --input_list ./input_list.txt
```

