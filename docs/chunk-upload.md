# 转写接口分片上传

服务设置 `ASR_CHUNK_UPLOAD_ENABLED=1` 后，继续通过 `POST /transcribe` 调用分片上传。默认值为 `0`。已有的整文件请求仍用 `action=transcribe` 和 `file`，请求和响应格式照旧。

| 顺序 | `action` | 表单字段 |
|---|---|---|
| 建立上传 | `upload_init` | `upload_filename`（例如 `meeting.wav`）、`upload_size`（原文件字节数）、`upload_sha256`（原文件 SHA-256） |
| 传单片 | `upload_part` | `upload_id`、`part_number`（从 1 开始）、`file`（该片原始字节） |
| 查询进度 | `upload_status` | `upload_id` |
| 合并 | `upload_complete` | `upload_id` |
| 发起整段转写 | `transcribe` | `upload_id` 和原有声纹、纠错等参数 |

初始化返回 `data.upload_id`、`part_bytes`、`part_count`、`expires_at`。每片默认 64 MiB，最后一片发送剩余字节。可乱序上传。相同片号和内容可以重复发送，内容不同返回 409。客户端重连后可查询 `data.received_parts` 并补传缺片。合并会检查整文件的字节数和 SHA-256，校验失败返回 422。`data.ready=true` 后提交转写。转写请求中只填写 `file` 或 `upload_id` 其中之一。此后的同步返回和异步纠错查询与整文件请求相同。

当分片开关关闭时，分片操作返回 HTTP 503，响应消息说明分片上传未启用。省略 `correct_text` 使用服务默认设置，`correct_text=true` 启用纠错，`correct_text=false` 跳过纠错。自动模式在等候预算内返回完成的纠错结果，超出预算则返回 HTTP 200、初稿和 `correction_task_id`，后台仍继续工作。可通过 `ASR_POSTPROCESS_AUTO_WAIT_SECONDS` 调整自动模式的预算，显式 `correction_mode=sync` 使用原同步等候时限。

上传文件上限沿用 `MAX_FILE_SIZE`。上传暂存默认位于声纹库所在目录的 `chunk_uploads` 子目录。可通过 `ASR_CHUNK_UPLOAD_DIR` 指定可写的持久卷目录。`ASR_CHUNK_UPLOAD_TTL_SECONDS` 默认 86,400 秒，`ASR_CHUNK_UPLOAD_CAPACITY_BYTES` 默认是 `MAX_FILE_SIZE` 的 4 倍，限制同时登记的原文件总大小。合并时需要额外暂存一份同等大小的文件。并行运行多个服务实例时，同一次上传的所有请求须访问同一份支持 SQLite 文件锁的存储，或定向到持有暂存卷的实例。

## 昇腾生产配置

生产转写启用昇腾提速和分片上传。声纹匹配使用 8 个工作线程，每个算子限制为单线程。自动纠错在初稿产生后等待最多 8 秒。

```dotenv
ASR_FAST_ENABLED=1
ASR_CHUNK_UPLOAD_ENABLED=1
ASR_POSTPROCESS_AUTO_WAIT_SECONDS=8
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMBA_NUM_THREADS=8
```
