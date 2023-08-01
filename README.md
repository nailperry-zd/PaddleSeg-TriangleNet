# TriangleNet: Edge Prior Augmented Network for Semantic Segmentation through Cross-Task Consistency
## Cityscapes

- The expriments were conducted using  single V100 GPU with batch size 4.

| Model       | Backbone  | Resolution | Training Iters | mIoU(val) | mIoU(test) | Links                                                        |
| ----------- | --------- | ---------- | -------------- | --------- | ---------- | ------------------------------------------------------------ |
| Baseline    | ResNet-18 | 1024x1024  | 300000         | 76.77     | 74.48      | [model](https://drive.google.com/file/d/10HvoCXtipagzGgyR6a5RVSatv_fVOKlA/view?usp=drive_link) |
| Trianglenet | ResNet-18 | 1024x1024  | 300000         | 78.96     | 77.36      | [model](https://drive.google.com/file/d/1cuba32NwBg5ke97-A5YAvDwNB8CAgLeG/view?usp=drive_link) |



### comparision with real-time semantic segmentation models

- All these inference speeds were measured using PaddleInference Api on a A100 GPU device. During this process, we use the PaddlePaddle implementations of these state-of-the-art models provided by [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) for fair comparision.

| Model         | Backbone | Test Resolution | mIoU(test) | FPS   |
| ------------- | -------- | --------------- | ---------- | ----- |
| ESPNetV2      | -        | 512x1024        | 66.2       | 126.5 |
| BiSeNetV1-L   | ResNet18 | 768x1536        | 74.7       | 83.9  |
| STDC1-Seg50   | STDC1    | 512x1024        | 71.9       | 262.1 |
| STDC2-Seg50   | STDC2    | 512x1024        | 73.4       | 207.4 |
| STDC1-Seg75   | STDC1    | 768x1536        | 75.3       | 152.7 |
| STDC2-Seg75   | STDC2    | 768x1536        | 76.8       | 131.5 |
| PP-LiteSeg-T1 | STDC1    | 512x1024        | 72.0       | 219.4 |
| PP-LiteSeg-B1 | STDC2    | 512x1024        | 73.9       | 184.3 |
| PP-LiteSeg-T2 | STDC1    | 768x1536        | 74.9       | 141.2 |
| PP-LiteSeg-B2 | STDC2    | 768x1536        | 77.5       | 118.4 |
| TriangleNet   | ResNet18 | 1024x2048       | 77.4       | 46.2  |



## FloodNet

- The expriments were conducted using  4 V100 GPUs with batch size 16.

| Model       | Backbone  | Resolution | Training Iters | mIoU(test) |
| ----------- | --------- | ---------- | -------------- | ---------- |
| Baseline    | ResNet-18 | 1024x1024  | 20000          | 65.64      |
| TriangleNet | ResNet-18 | 1024x1024  | 20000          | 70.97      |

