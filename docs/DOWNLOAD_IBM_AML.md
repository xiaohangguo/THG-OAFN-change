# IBM AML HI-Small 数据获取与审计

## 官方来源

- 数据与许可证说明：[IBM/AML-Data](https://github.com/IBM/AML-Data)
- 下载页：[IBM Transactions for Anti Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
- 数据集论文：Altman et al.，NeurIPS 2023。

先在 Kaggle 页面登录并接受条款。不要把 API token、下载的 CSV 或解压文件提交到本仓库。

## 目标目录

```text
D:\THG-OAFN-Financial-Experiment\data_cache\ibm_aml\
├── HI-Small_Trans.csv
└── HI-Small_accounts.csv
```

## 推荐下载方式

在已配置 Kaggle API token 的 PowerShell 中执行：

```powershell
python -m pip install kaggle
kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml `
  -p "D:\THG-OAFN-Financial-Experiment\data_cache\ibm_aml"
```

仅解压所需的 HI-Small 文件。完成后删除下载压缩包，避免重复占用空间。

## 审计门槛

```powershell
python scripts/phase1_data_audit.py `
  --transactions "D:\THG-OAFN-Financial-Experiment\data_cache\ibm_aml\HI-Small_Trans.csv" `
  --accounts "D:\THG-OAFN-Financial-Experiment\data_cache\ibm_aml\HI-Small_accounts.csv" `
  --output "D:\THG-OAFN-Financial-Experiment\metadata\ibm_hi_small_audit.json" `
  --hash
```

审计应报告时间范围、正样本率、账户数量、银行数量、字段映射和 SHA-256。任何缺失时间戳、未知标签值或缺失交易端点都会以失败退出，不能绕过。
