"""
主程式 - 圖像生成模型執行入口點
"""

import os
import argparse
from modules.config import BATCH_SIZE, MODES
from modules.data_utils import create_dataset_df, load_and_preprocess_image, normalize_boxes, create_output_dir
from modules.model_utils import setup_models, generate_description, clean_memory
from modules.image_utils import generate_image, save_image

def process_batch(batch_df, images_dir, output_dir, processor, blip_model, pipe, prompt_type, use_layout):
    """
    處理一個批次的圖像
    
    Args:
        batch_df: 批次數據DataFrame
        images_dir: 輸入圖像目錄
        output_dir: 輸出圖像目錄
        processor: BLIP處理器
        blip_model: BLIP模型
        pipe: 擴散模型管道
        prompt_type: 提示詞類型
        use_layout: 是否使用佈局信息
    """
    for _, row in batch_df.iterrows():
        image_path = os.path.join(images_dir, row['image_filename'])
        try:
            # 載入並預處理圖像
            processed_image = load_and_preprocess_image(image_path)
            
            # 生成圖像描述
            description = generate_description(processed_image, processor, blip_model, prompt_type)
            
            # 生成圖像
            if use_layout:
                normalized_boxes = normalize_boxes(
                    row['bboxes'],
                    row['width'],
                    row['height']
                )
                generated_image = generate_image(
                    pipe,
                    description,
                    use_layout=True,
                    boxes=normalized_boxes,
                    phrases=row['labels']
                )
            else:
                generated_image = generate_image(
                    pipe,
                    description,
                    use_layout=False
                )
            
            # 保存生成的圖像
            output_path = os.path.join(output_dir, row['image_filename'])
            save_image(generated_image, output_path)
            print(f"處理並保存: {row['image_filename']}")
            
            # 清理記憶體
            del processed_image, description
            if use_layout:
                del normalized_boxes
            del generated_image
            clean_memory()
            
        except Exception as e:
            print(f"處理 {image_path} 時發生錯誤: {str(e)}")

def run_generation(mode, images_dir, json_path, output_dir=None, batch_size=None):
    """
    運行圖像生成
    
    Args:
        mode: 生成模式 (從MODES中選擇)
        images_dir: 輸入圖像目錄
        json_path: 標籤文件路徑
        output_dir: 自訂輸出目錄 (可選)
        batch_size: 自訂批次大小 (可選)
    """
    if mode not in MODES:
        raise ValueError(f"不支援的模式: {mode}。支援的模式有: {', '.join(MODES.keys())}")
    
    # 讀取模式配置
    mode_config = MODES[mode]
    
    # 設置輸出目錄
    if output_dir is None:
        output_dir = mode_config["output_dir"]
    
    # 創建輸出目錄
    create_output_dir(output_dir)
    
    # 設置批次大小
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    # 創建數據集
    dataset_df = create_dataset_df(json_path)
    
    # 設置模型
    processor, blip_model, pipe = setup_models(
        mode_config["description_model"],
        mode_config["generation_model"]
    )
    
    # 分批處理數據
    for i in range(0, len(dataset_df), batch_size):
        batch_df = dataset_df.iloc[i:i+batch_size]
        process_batch(
            batch_df, 
            images_dir, 
            output_dir, 
            processor, 
            blip_model, 
            pipe, 
            mode_config["prompt_type"],
            mode_config["use_layout"]
        )
        
        print(f"完成批次 {i//batch_size + 1}/{(len(dataset_df) + batch_size - 1)//batch_size}")

def main():
    # 設置命令行參數解析器
    parser = argparse.ArgumentParser(description="圖像生成工具")
    parser.add_argument(
        "mode", 
        choices=MODES.keys(),
        help="生成模式"
    )
    parser.add_argument(
        "--images_dir", 
        default="images",
        help="輸入圖像目錄路徑"
    )
    parser.add_argument(
        "--json_path", 
        default="label.json",
        help="標籤文件路徑"
    )
    parser.add_argument(
        "--output_dir", 
        default=None,
        help="輸出目錄路徑 (可選，預設使用模式對應的輸出目錄)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int,
        default=None,
        help="批次大小 (可選，預設使用配置中的批次大小)"
    )
    
    # 解析命令行參數
    args = parser.parse_args()
    
    # 運行生成
    run_generation(
        args.mode,
        args.images_dir,
        args.json_path,
        args.output_dir,
        args.batch_size
    )

if __name__ == "__main__":
    main() 