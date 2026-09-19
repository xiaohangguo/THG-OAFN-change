# IBM AML HI-Small 数据获取与审计

## 官方来源

- 数据与许可证说明：[IBM/AML-Data](https://github.com/IBM/AML-Data)
- 下载页：[IBM Transactions for Anti Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
- 数据集论文：Altman et al.，NeurIPS 2023。

先在 Kaggle 页面登录并接受条款。不要把 API token、下载的 CSV 或解压文件提交到本仓库。

## 目标目录

```text
D:\THG-OAFN-Financial-Experiment\
├── secrets\
│   └── kaggle.json          # Kaggle API token（不进 Git）
├── data_cache\ibm_aml\
│   ├── HI-Small_Trans.csv
│   ├── HI-Small_accounts.csv
│   └── HI-Small_Patterns.txt
└── metadata\
    └── ibm_hi_small_audit.json
```

## Kaggle 凭据（放在 D 盘）

从 Kaggle Settings → API → Create New Token 下载 `kaggle.json`，放入
`D:\THG-OAFN-Financial-Experiment\secrets\kaggle.json`，然后在 PowerShell 中导出：

```powershell
$env:KAGGLE_CONFIG_DIR = "D:\THG-OAFN-Financial-Experiment\secrets"
```

## 推荐下载方式（白名单脚本，只下 HI-Small 三个文件）

```powershell
python -m pip install kagglehub
$env:KAGGLE_CONFIG_DIR = "D:\THG-OAFN-Financial-Experiment\secrets"
python scripts/download_ibm_aml.py
```

脚本只下载 `HI-Small_Trans.csv`（约 476 MB）、`HI-Small_accounts.csv`（约 34 MB）、
`HI-Small_Patterns.txt`，绝不会触碰约 42 GB 的 HI-Large 包；已存在的文件自动跳过。

不要使用 `kaggle datasets download -d ealtman2019/...` 不带文件名的形式，
那会把整个数据集（含 HI-Large）全部拉下。

## 网络兜底

本机实测 kaggle.com 与 storage.googleapis.com 可达。若下载超时，先走本地代理：

```powershell
$env:HTTPS_PROXY = "http://127.0.0.1:7897"
```

仍失败时用浏览器在 Kaggle 数据集页手动下载三个 HI-Small 文件，
放入 `D:\THG-OAFN-Financial-Experiment\data_cache\ibm_aml\` 即可，审计器不关心获取方式。

## 审计门槛

```powershell
python scripts/phase1_data_audit.py `
  --transactions "D:\THG-OAFN-Financial-Experiment\data_cache\ibm_aml\HI-Small_Trans.csv" `
  --accounts "D:\THG-OAFN-Financial-Experiment\data_cache\ibm_aml\HI-Small_accounts.csv" `
  --output "D:\THG-OAFN-Financial-Experiment\metadata\ibm_hi_small_audit.json" `
  --hash
```

审计应报告时间范围、正样本率、账户数量、银行数量、字段映射和 SHA-256。任何缺失时间戳、未知标签值或缺失交易端点都会以失败退出，不能绕过。
