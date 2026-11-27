# A6API / Qwen CoT 评估与本地核查网页

入口文件名保留为 `evaluate_cot_glm.py`。当前默认使用 **A6API + `qwen3.8-max`**，同时保留 `--provider zhipu` 兼容旧 GLM。Python 3.10+，只需要 Pillow；现有 `IAD-VLMs` 环境已具备依赖。脚本直接发送与 OpenAI SDK 等价的 HTTPS Chat Completions 请求，不需要额外安装 SDK 或模型。

| 配置 | 当前默认值 |
| --- | --- |
| provider | `a6api` |
| Base URL | `https://api.a6api.vip/v1` |
| 完整接口 | `https://api.a6api.vip/v1/chat/completions` |
| model | `qwen3.8-max`（实际模型 ID，与令牌名称无关） |
| 评估输出 | `concise`：两个维度各一句判断与依据，合计最多 100 字，不打分 |
| 输出 token 上限 | `512`（JSON 字段也占 token，token 数与中文字数不是同一单位） |
| 凭证环境变量 | `A6API_KEY` |
| 输出目录 | `data/cot_evaluation_a6api_qwen38max/` |

根据 [A6API 开发文档](https://a6api.vip/#/docs)，网关兼容 OpenAI 请求和 Bearer 认证。模型及可选参数能否使用取决于当前令牌和上游线路；文档没有保证 `qwen3.8-max` 每条线路都支持图片。因此先使用下面的两图检查，再开始评估。纯文字测试成功不能证明视觉能力。

## 快速运行

以下命令在仓库根目录运行：

```bash
cd /mnt/data/wangzijian/second_stage_work/code/IAD-VLMs
conda activate IAD-VLMs

# 只需在终端填入 Key，无须修改或提交源代码。
export A6API_KEY='在这里替换为A6API的完整_API_Key'

# 使用两张随机色块图片检查认证、模型和多图读取；不读取数据集，也不伪造评分。
python scripts/00_generate/anomaly_instruct_4/evaluate_cot_glm.py check-api

# 先试跑 8 条；检查图像与评语是否符合预期。
python scripts/00_generate/anomaly_instruct_4/evaluate_cot_glm.py evaluate --model qwen3.8-max --workers 1 --limit 8

# 全量评估；同配置下会跳过已完成条目，重试错误条目。
python scripts/00_generate/anomaly_instruct_4/evaluate_cot_glm.py evaluate --model qwen3.8-max --workers 1 --request-interval 1 --retries 8
```

Key 在 `get_api_key()` 函数读取。A6API 只读 `A6API_KEY`，不会把旧的智谱 Key 自动发送给新服务；如需不同环境变量可显式设置 `--api-key-env`。不要把真实 Key 写进 Python、网页或结果文件。未设置 Key 时会提示并退出。

`check-api` 会让模型读取两张随机生成的色块图，并核对两张图中各色块的顺序；结果保存在 `api_check.json`。通过表示本次线路确实完成了这个简单多图任务，不保证工业缺陷判断的精度。失败时先确认令牌、模型 ID 和线路的图片输入能力；只有文字连通时不要直接批量评视觉忠实度。此检查发送一次小请求，实际调用由你提供 Key 后执行。

在另一个终端启动网页，**不需要 API Key，也不需要先完成评估**：

```bash
cd /mnt/data/wangzijian/second_stage_work/code/IAD-VLMs
conda activate IAD-VLMs
python scripts/00_generate/anomaly_instruct_4/evaluate_cot_glm.py serve --port 8765
```

在运行脚本的同一台机器访问 **http://127.0.0.1:8765**。端口占用时可改为 `--port 8766`。两个进程共用 SQLite，网页每 30 秒更新概况和当前评估，也可点击“刷新数据”。网页入口会先核对源文件并更新索引，终端显示 URL 后即可访问。Ctrl+C 关闭网页。

如果脚本运行在远程服务器，浏览器在你的个人电脑，请在**个人电脑的终端**运行：

```bash
ssh -N -L 8765:127.0.0.1:8765 你的用户名@服务器地址
```

保持 SSH 连接后，在个人电脑访问 http://127.0.0.1:8765。不需要将服务监听改为公网地址。

没有激活 conda 时，可以把以上 `python` 替换为：

```text
/mnt/data/wangzijian/miniconda3/envs/IAD-VLMs/bin/python
```

## 数据范围和存储

默认输入：`data/traces_qwen3vl8b_evidence_first/traces_{dataset}.json` 和对应 `data/manifests/{dataset}.json`。按 dataset + image_id 取最后一条记录，仅接收 `validation.valid == true` 且有完整 reasoning 的样本，并排除 `failed/cot_retry_2.jsonl` 中的样本。这里的 valid 是**原生成程序的结构校验通过**，不表示视觉描述已被证实。

2026-11-05 核对结果：

| 数据集 | 成功 CoT | manifest 中无成功 CoT |
| --- | ---: | ---: |
| anomaly_shapenet | 1,923 | 1 |
| mvtec3d | 3,842 | 1 |
| mvtec_ad | 5,349 | 5 |
| real3d | 1,254 | 0 |
| 合计 | **12,368** | **7** |

新接口结果默认放在 **`data/cot_evaluation_a6api_qwen38max/`**，也可以通过 `--output-dir` 指定独立实验目录。旧的 `data/cot_evaluation_glm46v/` 保持原样，使用 `--provider zhipu` 时默认读取旧目录：

| 文件 | 内容 |
| --- | --- |
| `evaluations.sqlite3` | 实时主数据：规范化样本索引、去重路径、评估结果、每次请求的响应/错误/耗时、人工核查记录 |
| `evaluations_{dataset}.jsonl` | 四个数据集各一份当前配置的紧凑评估结果；包含 image_id、fingerprint、profile、status、evaluation、error |
| `summary.json` | 各数据集评估数量、通过/复核/不通过数量、两个维度的均分与输入统计 |
| `evaluation_profile.json` | 模型、接口、完整评分提示词、采样参数、图像设置，用于追溯 |
| `api_check.json` | 两张随机图的 API 检查结果，不含 API Key 或测试图 Base64 |
| `evaluate.lock` | 防止同一输出目录被两个评估进程同时重复调用 |

数据库逐次请求提交；JSONL 和 summary 在评估结束或正常 Ctrl+C 后导出。评估过程中如需最新导出，可另开终端执行 `export`。只初始化网页或索引时不会虚构评分，也不会生成模拟评估结果。

批量请求的完整响应在数据库 `batch_attempts` 表中每次只保存一份，包含整批 usage、耗时及当前请求的样本编号映射；单样本 `attempts.response` 保存 `batch_attempt_id` 引用，不重复保存整批响应。批次协议提示保存在 `meta['multi-sample-v1']`，原始请求里的图片 Base64 和 API Key 均不落盘。整批 token 用量不要按引用它的样本行重复求和。

**不复制图片，不复制原 CoT，不保存请求中的 Base64。** 数据库 `assets` 表对图片路径去重，每个样本通过资产 ID 引用；`sources` 表对原始 JSON 路径去重。CoT 按需从原 JSON 读取，带框图用原图与现有 GT 坐标在内存中绘制，不在磁盘生成一批重复图片。原始文件路径需继续可读，不应在评估中途移动或修改源文件。

## 评估输入与方法

每个样本独立得到两个维度的结果；默认每次请求一条，也可通过批量模式在一个请求里评估多条。输入包括：

1. 无标注 query 和按视角编号配对的正常 reference。具体比较以同一物理部位是否可见为准。二维样本 2 张，多视角样本通常 8 张。
2. 物体类别、原问题、完整候选 `reasoning`，含 `<think>`、`<location>`（兼容 `<locate>`）、`<type>`、`<answer>`。正常样本缺少 location/type 属于正常情况。
3. 明确的图片角色、view_id、原始尺寸及发送尺寸。坐标始终解释为原始图片 inclusive pixel xyxy。

GT 标签和类型、mask、原始路径、样本 ID、GT 衍生的源描述和 `evidence_review` 不写入评审请求。**候选 CoT 自身的 type/answer/location 属于待核查声明**。

默认 A6API 精简评审按样本选择输入。正常样本和无框异常样本使用无框 query 和正常 reference。有真实标注框的异常样本还会随对应视角附上带框副本。模型用框定位，再回到原图核对候选的局部描述，推理仍按原标准审查。该评审使用了标注信息，适用于 CoT 质检。所有结果写入同一目录，带框评审结果在现有 JSON 内标记 `region_review: true`。

| 维度 | 检查项 |
| --- | --- |
| `visual_faithfulness` 视觉忠实度 | 图像证据、位置、颜色、形状；包含正常描述、异常描述与多视角一致性 |
| `reasoning_consistency` 推理合理性 | 观察到推断、推断到结论、标签一致性、不确定性表达 |

### 当前默认：100 字精简判断

默认 `--evaluation-format concise`，模型输出两段简短判断：

```json
{"visual_faithfulness":"忠实：绿色表面与完整轮廓有图像支持。","reasoning_consistency":"合理：观察与正常结论一致，未推断不可见结构。"}
```

视觉判断为“忠实 / 不忠实 / 不确定”，推理判断为“合理 / 不合理 / 不确定”。两项正文（包括判断及标点）合计最多 100 字符。视觉判“不忠实”时，响应额外带 `visual_conflict`，记录查询视角、候选原文和直接矛盾的依据类别。引用缺失、原文或视角不符、或依据只有参考差异时，程序把视觉结论改为“不确定”。该记录用于追溯，程序可检查引用，图像判断仍需要复核。默认请求 `max_tokens=512`。

程序拒绝空正文或缺少判断依据的回复。若格式正确但轻微超长，会在本地将两段正文限长到合计 100 字，保留明确判断并标记 `length_limited`；不为缩短回复重复付费请求。完整原始响应仍保存在 `attempts` 中，可追溯。程序添加的 `judgments`、`text_chars`、`verdict` 等属于本地元信息，不是额外要求模型生成的内容。

筛选状态由判断生成：任一维度“不忠实/不合理”为不通过；其余只要有“不确定”为需复核；两者都获支持为通过。两个维度独立评价，视觉事实错误不代表内部逻辑必然错误。

在服务、模型、采样配置、图片缩放及样本 fingerprint 全部相同的情况下，单纯切换到精简格式可以原样沿用成功历史评分，并标明来源。需要把指定旧样本重新评为简短判断时，可使用 `--image-id ... --force`；省略样本限制会重评整个筛选范围。历史判断不会被自动改写成新的模型结论。

### 历史评分版

`--evaluation-format scored` 可保留旧版输出，默认输出上限为 16384 tokens。每个维度有 1–5 整数分、总评依据和逐项 verdict/reason。逐项 verdict 为 `supported`（支持）、`contradicted`（矛盾）、`uncertain`（不能确定）、`not_applicable`（无对应声明）。

5 表示清晰充分；4 基本可靠，只有轻微问题；3 混合支持、实质疑点或推导跳步；2 关键声明不受支持或主要推导矛盾；1 核心与图像或自身逻辑矛盾。视觉事实错误与内部推导错误分别判断，不机械同步扣分。

`issues` 记录问题维度、严重性、候选文本的精确原文摘录、涉及的 query 视角和解释。`confidence` 表示评估模型对本次判断的把握。程序校验必需字段、分数范围、问题摘录确实存在于原文、view_id 确实存在；拒绝被截断、被过滤、空 final content 或格式错误的回复，不用 `reasoning_content` 代替评估 JSON。

程序生成网页筛选用的 `verdict`：

- `fail` 不通过：任一维度 ≤2，或存在 critical 问题。
- `review` 需复核：未达到 fail，且任一维度 ≤3、confidence 为 low、检查项存在 uncertain/contradicted，或 issues 非空。
- `pass` 通过：其余情况。

这些是**模型评审意见与筛选阈值**；“通过”仍可人工核查。输入分辨率、视角和模型感知能力会影响结果。修改评分规则或参数后应视为新的评估配置。

## 可续跑与限速

默认两个并发请求，全局每秒最多启动一个请求；实际速度还取决于模型延迟。遇到 HTTP 429 或 1302/1305 等速率限制/模型繁忙时，**所有线程共享冷却时间**：默认约 30、60、120、240 秒，之后上限约 300 秒，另有少量随机延迟。服务端提供 Retry-After 时至少等待它指定的时长。冷却也作用于重试次数用完后的下一个样本，防止整批任务持续冲击繁忙服务；成功获得有效评估后退避级数重置。用 `--rate-limit-wait` 调整初始等待秒数。

网络超时、服务错误和格式错误仍使用较短的指数退避；默认最多 3 次额外尝试，可通过 `--retries` 调整。认证、权限、无效参数、欠费或明确的配额用尽会停止批次；HTTP 429 会根据业务码区分模型繁忙与账户/套餐限制。每次尝试保留响应、错误和 token usage 便于排查。业务码依据[官方错误码说明](https://docs.bigmodel.cn/cn/api/api-code)。

Ctrl+C 会停止安排新样本，等待已经发出的请求结束或超时，然后导出。重新运行同一命令会跳过成功结果，重试失败与未评估样本。图片元信息或样本内容改变时，旧结果不会被当成当前结果展示。prompt、模型、thinking、temperature、token 上限或图片缩放配置改变时，会使用新的 profile，历史尝试仍在数据库中。并发数、限速、重试、冷却时间和 limit 不影响 profile。

### 多请求并发与单请求多样本

这两种设置可以叠加：

| 参数 | 意义 |
| --- | --- |
| `--workers 2` | 同时最多发送 2 个独立 API 请求 |
| `--batch-size 4` | 一个请求中最多包含 4 条样本；默认 1，保持单样本模式 |
| `--batch-max-images 32` | 每个请求的图片预算，默认 32；包含带框副本，超出时自动拆批，保留全部视角 |
| `--max-tokens 512` | 每条样本的输出 token 预算，实际请求乘以本批样本数，例如 4 条为 2048 |

`batch-size > 1` 只支持 `--provider a6api --evaluation-format concise`。这是在标准 Chat Completions 中拼入多个独立样本，并用 `results` 数组收回结果；不依赖 A6API 的原生异步 Batch 端点，也不代表上游模型内部并行处理。`workers` 才是本脚本的请求并发数。

一个无框多视角样本通常有 8 张图片，二维样本有 2 张；每个有框视角增加 1 张带框副本。同一请求的图片数、样本数和本地输出 token 总预算（32768）任一达到上限都会拆批，实际每批可少于 `--batch-size`。这个图片预算不是对具体模型或线路能力的保证。

建议先用 2–4 条的小批检查编号与内容，批次太大可能增加跨样本混淆。批量可减少请求次数和重复提示开销，但查询图、CoT 和各样本的输出 token 仍会消耗，费用不一定按批量大小等比例下降，最终以平台日志为准。

```bash
# 小规模验证：本次最多评估 8 条待完成样本。
CUDA_VISIBLE_DEVICES=1 python scripts/00_generate/anomaly_instruct_4/evaluate_cot_glm.py evaluate \
  --model qwen3.8-max --evaluation-format concise \
  --batch-size 4 --workers 2 --max-tokens 512 --timeout 300 --limit 8

# 检查结果后全量运行；已成功样本自动跳过。
CUDA_VISIBLE_DEVICES=1 python scripts/00_generate/anomaly_instruct_4/evaluate_cot_glm.py evaluate \
  --model qwen3.8-max --evaluation-format concise \
  --batch-size 4 --workers 2 --max-tokens 512 --timeout 300
```

每个样本由 `SAMPLE_BEGIN / SAMPLE_END` 分界，API 只看到中性的请求内 `sample_index`，不提供可能暴露缺陷标签的原始样本 ID 或路径。每条样本仍独立要求两项正文合计最多 100 字。解析时按编号对应，允许结果乱序；缺失、重复或格式错误的样本会单独重试，正确结果先提交，不重新发送已成功的邻近样本。未知编号导致整份回复无法可靠对应，会拒绝本次回复。坏图片只影响所属样本。429 冷却与账户错误停止策略对全部批次线程共享。

`batch-size`、图片批预算和 `workers` 只改变请求安排，不改变单样本评估任务的 profile；切换单条/批量模式不会自动重复评估已成功样本。每次批量请求的实际成员、编号、图片数、输出上限及协议版本在 `batch_attempts.request_info` 留存，便于追溯。

先少量检查后再全量运行。并发额度以你账号实际返回为准，可减小参数：

```bash
python scripts/00_generate/anomaly_instruct_4/evaluate_cot_glm.py evaluate --workers 1 --request-interval 2

# 1305 模型繁忙时建议使用单并发、更长间隔和更多等待重试。
python scripts/00_generate/anomaly_instruct_4/evaluate_cot_glm.py evaluate --workers 1 --request-interval 3 --rate-limit-wait 30 --retries 8

# 只评一个数据集 / 一个样本
python scripts/00_generate/anomaly_instruct_4/evaluate_cot_glm.py evaluate --dataset mvtec_ad
python scripts/00_generate/anomaly_instruct_4/evaluate_cot_glm.py evaluate --image-id 'mvtec_ad__bottle_test_broken_large_000.png'

# 强制重评指定样本。省略 --image-id 会重评筛选范围内所有成功样本。
python scripts/00_generate/anomaly_instruct_4/evaluate_cot_glm.py evaluate --image-id 'mvtec_ad__bottle_test_broken_large_000.png' --force

# 不调用 API：检查索引、查看汇总、导出当前结果
python scripts/00_generate/anomaly_instruct_4/evaluate_cot_glm.py prepare
python scripts/00_generate/anomaly_instruct_4/evaluate_cot_glm.py summary
python scripts/00_generate/anomaly_instruct_4/evaluate_cot_glm.py export
```

如需在其他目录的数据上工作，`--trace-dir`、`--manifest-dir`、`--output-dir` 应在评估和网页命令中保持一致。网页只显示该目录的 active profile，并显示当前模型与服务名。默认模型为 `qwen3.8-max`；`--model` 可以显式覆盖，`--base-url https://api.a6api.vip/v1` 自动追加 `/chat/completions`，也可用 `--endpoint` 填完整 URL，两者不能同时使用。接口要求 HTTPS。服务或模型变化会使用独立 profile，不将旧 GLM 评分当作 Qwen 评分；同一配置续跑才跳过成功记录。

旧 GLM 仍可通过以下命令运行或查看，凭证为 `ZHIPU_API_KEY`（兼容 `GLM_API_KEY`）。旧 GLM profile 的计算方式保持兼容：

```bash
python scripts/00_generate/anomaly_instruct_4/evaluate_cot_glm.py evaluate --provider zhipu
python scripts/00_generate/anomaly_instruct_4/evaluate_cot_glm.py serve --provider zhipu --port 8766
```

默认保留原图像素；超过 API 尺寸/字节限制时才在内存中转换，不修改原图。必要时可显式设置 `--max-image-side 1024`，这会创建新 profile。缩图时请求中同时标明原始与发送尺寸，候选框坐标仍按原图解释。

## 网页操作

页面采用本地无衬线字体，正文和 CoT 提高字号与字重，标题、类别、统计和评估判断加粗。淡蓝色粒子连线背景仅位于内容后方，可在页脚切换“背景动画”；偏好保存在当前浏览器。打开详情弹窗、切到后台或系统启用“减少动态效果”时暂停动画，不影响样本核查。

主页是图片卡片图库，按数据集、类别、GT 正常/异常、异常类型、自动评估状态、两个维度的判断、人工核查状态筛选。支持按 ID/评语搜索、待复核优先排序、每页 30/50/100 条及页码跳转。缩略图在内存中生成并按页延迟加载，不在磁盘复制图片，也不一次性读取万条 CoT。

点击卡片图片打开样本详情弹窗，内有全视角切换、query 原图、query + GT 框图、正常参考图、完整 CoT 标签、原生成记录及 evidence_review、两个维度的简短判断和评估 JSON。右上角 ×、Esc 或点击遮罩关闭弹窗，保留主页筛选、页码和浏览位置，继续查看其他样本。在弹窗中按 J/K 切换样本、左右方向键切换视角，点击详情图片可再次放大。历史评分版仍按原内容显示，并保留问题原文定位功能。

人工记录支持“已核查”“待跟进”和备注；点击“保存记录”才持久化到 SQLite。选择“未核查”并清空备注再保存，会清除该样本记录。未保存的编辑在当前页面切换样本时保留为内存草稿，关闭或刷新页面会丢失。人工记录绑定样本 fingerprint，输入变化后旧记录不再作为当前核查记录显示。

## 验证与官方依据

离线测试不访问任何外部模型 API：

```bash
python -m unittest discover -s scripts/00_generate/anomaly_instruct_4 -p test_cot_evaluation.py -v
node --check scripts/00_generate/anomaly_instruct_4/cot_review_web/app.js
```

当前 A6API 适配依据 [开发文档](https://a6api.vip/#/docs)：使用 `/v1/chat/completions`、`Authorization: Bearer`，从最终 `choices[0].message.content` 解析结果。多图以 OpenAI 兼容的 `image_url.url=data:image/png;base64,...` 或 `data:image/jpeg;base64,...` 发送，保留实际 MIME。默认不发送 GLM 专用 `thinking`、`enable_thinking` 或未确认支持的 JSON Schema 参数；由相同评分提示词约束 JSON，再在本地检查。请求 ID 从 `X-Oneapi-Request-Id` 留存到响应或错误日志，方便定位问题。`qwen3.8-max` 是用户指定的模型 ID，实际图片能力由用户的令牌和线路实测确认，不根据模型名称推断。

脚本沿用每张小于 5 MB、最长边不超 6000、单次最多 50 张的保守本地预算；这些不是 A6API 对所有模型的能力承诺。精简版默认 `max_tokens=512`，历史评分版默认 16384，脚本参数上限 32768；实际可用 token 范围仍以当前上游模型为准，可以通过 `--max-tokens` 调整。

旧智谱适配依据 [GLM-4.6V-Flash 官方指南](https://docs.bigmodel.cn/cn/guide/models/free/glm-4.6v-flash) 与 [视觉模型 API 定义](https://docs.bigmodel.cn/api-reference/模型-api/对话补全)，继续使用原来的裸 Base64 与 `thinking.type`。两种适配均只从最终回答解析评估，`reasoning_content` 留存为原始响应，不代替评估 JSON。
