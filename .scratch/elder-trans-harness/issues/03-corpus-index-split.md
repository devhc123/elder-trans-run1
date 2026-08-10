# 03 — 语料索引与 record_id 三分切

**What to build:** 把 `/Users/chenhao/DATA/03_国内医疗语料与数据源/2025国内医疗爬虫结果/` 下 2.8G、七个来源子库建成一份可查询的索引，并按 `record_id` 切成互不相交的三份：测试集池、verifier 训练池、备用扩样池。

七库：bdyd（百度医典，疾病 65 万 / 检查 1.67 万带「报告解读」字段 / 药品）、xywy（寻医问药 13 万，带 medicalInsurance、clinicalDepartment、treatmentCycle）、zsys（9.3 万）、others（diseaseKg 8.9 万带 yibao_status/do_eat/not_eat + drug_info 1.77 万带禁忌/相互作用/警示）、ylys（药品 2.2 万带同类药对比 / 指标 2.97 万 / 饮食指导 1670 带宜吃少吃慎吃）、dxys（丁香医生）、mkss。

**工程上最麻烦的是编码**：多个文件名是 GBK 乱码（`ҩƷ.csv`=药品、`Ѩλ.csv`=穴位、`֢״.csv`=症状），部分 CSV 内容非 UTF-8。索引必须把这些正确还原，否则后面全链路带病。

**切分必须在 `record_id` 层做，不能在题目层做。** 同一条源记录可以派生多道题——同类项目的教训是它 108 道题只对应 85 个 `record_id`，而它 7387 条候选池里有 166 条与测试集 `record_id` 重合，属于事实上的泄漏。切分种子固定并记录，交集为空要能程序化验证。

**Blocked by:** None — can start immediately.

**Status:** done — `pipeline/source_index.py` + `pipeline/split_records.py`

- [x] 七库全部建索引，每条记录有稳定的 `record_id`（形如 `<库>_<逻辑表>#row<N>`）与字段清单
- [x] GBK 乱码文件名与非 UTF-8 内容正确还原，索引里是可读中文
- [x] 三分切产出 test / verifier-train / 备用三个 `record_id` 集合，**两两交集为空**，有脚本可验证
- [x] 切分种子写进 kpi.yaml，重跑得到相同切分
- [x] 索引可按库、按字段、按老年高发病/常用药筛选，供下游抽样使用


**实测结果：** 83,730 条逻辑记录；test 16,614 / verifier_train 50,370 / reserve 16,746，两两交集为 0。

**一个必须记住的修正：** 早先的数据盘点报告说 bdyd/疾病 有 651,924 条 —— 那是**物理行数**，字段内含换行。真实逻辑记录是 4,273 条。全库真实规模 83,730 条，不是「65 万疾病」。凡引用语料规模一律用记录数，已写进 kpi.yaml 的 caveats 并加了回归测试。

**文件名乱码的处理取向：** 反解 `žąūČ` 会得到 `啪膮奴膶` —— 全是汉字、看着像模像样，实际是错的。这种"貌似成功"的降级比直接失败危险，所以逻辑表名走显式映射，每条由打开文件读实际记录认定（样本证据写在 `LOGICAL_NAMES` 的注释里），未认定的乱码名直接报错拒绝猜。
