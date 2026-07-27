# Thpolish 模型文件目录

请将试剂体系对应的模型文件放入本目录：

| 文件 | 说明 |
|------|------|
| `S1.filter.v1.0.pkl` | LightGBM 候选位点筛选模型（S1 体系） |
| `S1.predict.v1.0.pt` | PyTorch 碱基校正预测模型（S1 体系） |

模型文件体积较大，不纳入 git 管理，请从本仓库的
[Releases](../../../../releases) 页面下载后放入本目录。

其他试剂体系的模型命名规则为：`<REAGENT>.filter.v1.0.pkl` 和
`<REAGENT>.predict.v1.0.pt`，使用时通过 `cycasm.sh -r <REAGENT>` 选择。
