# CVPDL HW 3: Data Augmentation

## 架構

```
Codes for inference/
├── modules/                  # 模組目錄
│   ├── __init__.py           # 模組初始化文件
│   ├── config.py             # 配置參數
│   ├── data_utils.py         # 數據處理相關函數
│   ├── model_utils.py        # 模型載入和設定相關函數
│   └── image_utils.py        # 圖像生成和處理相關函數
├── run_generation.py         # 主程式入口點
```

## 程式運行

```bash
python Codes\ for\ inference/run_generation.py <模式> [選項]
```

支援的模式：
- `no_layout_prompt1`: 使用 BLIP 2.7B 模型生成簡單圖像描述，使用 Stable Diffusion 生成圖像
- `no_layout_prompt1_model2`: 使用 BLIP 2.7B 模型生成詳細的工作場所描述，使用 Stable Diffusion 生成圖像
- `no_layout_prompt2_model2`: 使用 BLIP 6.7B 模型生成詳細的工作場所描述，使用 Stable Diffusion 生成圖像
- `prompt2_model2_with_layout`: 使用 BLIP 6.7B 模型生成詳細的工作場所描述，結合 GLIGEN 模型使用 layout 信息生成圖像

選項：
- `--images_dir`: 輸入圖像目錄路徑（預設: `images`）
- `--json_path`: 標籤文件路徑（預設: `label.json`）
- `--output_dir`: 輸出目錄路徑（可選，預設使用模式對應的輸出目錄）
- `--batch_size`: 批次大小（可選，預設使用配置中的批次大小）

### 範例

1. 運行不含 layout 的圖像生成（Prompt 1, BLIP 2.7B）：
   ```bash
   python Codes\ for\ inference/run_generation.py no_layout_prompt1
   ```

2. 運行不含 layout 的圖像生成（Prompt 2, BLIP 2.7B）：
   ```bash
   python Codes\ for\ inference/run_generation.py no_layout_prompt1_model2
   ```

3. 運行不含 layout 的圖像生成（Prompt 2, BLIP 6.7B）：
   ```bash
   python Codes\ for\ inference/run_generation.py no_layout_prompt2_model2
   ```

4. 運行含 layout 的圖像生成（使用 GLIGEN 模型）：
   ```bash
   python Codes\ for\ inference/run_generation.py prompt2_model2_with_layout
   ```

5. 指定自訂輸入和輸出目錄：
   ```bash
   python Codes\ for\ inference/run_generation.py no_layout_prompt1 --images_dir my_images --json_path my_labels.json --output_dir my_output
   ```

## 輸入與輸出

- **輸入數據**：
  - 請將包含圖像的資料夾與標籤檔案放置在執行目錄下：
    ```
    ./images/
    ./label.json
    ```

- **輸出**：
  - 生成的圖像將保存在以下目錄（除非通過 `--output_dir` 指定自訂目錄）：
    ```
    ./no_layout_prompt1/          # 運行 no_layout_prompt1 模式
    ./no_layout_prompt1_model2/   # 運行 no_layout_prompt1_model2 模式
    ./no_layout_prompt2_model2/   # 運行 no_layout_prompt2_model2 模式
    ./prompt2_model2_with_layout/ # 運行 prompt2_model2_with_layout 模式
    ```