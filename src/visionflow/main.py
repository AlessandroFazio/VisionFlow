import PIL
import PIL.Image
import yaml

from visionflow.core.visionflow import VisionFlow, VisionFlowConfig


def main():
    config_path = ""
    with open(config_path) as f:
        content = yaml.safe_load(f)
    
    img_path = ""
    img_bytes = PIL.Image.open(img_path).tobytes()
    
    config = VisionFlowConfig.model_validate(content)
    vf = VisionFlow(config)
    
    pipeline = (
        vf.builder("pokerstars_recognizer")
          .detect("poker/table_detection")
          .split_by_detections()
            
            .branch("card")
              .filter(min_conf=0.70)
              .crop_to_detection()
              .classify("poker/card_classifier")
              .apply(lambda ex:ex)
            .end_branch()
            
            .branch("player_info")
              .filter(min_conf=0.70)
              .crop_to_detection()
              .resize((2,2))
              .binarize()
              .ocr("poker/player_info_ocr")
            .end_branch()
            
            .branch("player_in")
              .filter(min_conf=0.70)
            .end_branch()
            
            .branch("dealer_button")
              .filter(min_conf=0.70)
            .end_branch()
            
            .branch("chips_amount")
              .filter(min_conf=0.70)
              .crop_to_detection()
              .detect("poker/chips_detector")
              .split_by_detections()
                .branch("chips")
                  .filter(min_conf=0.70)
                  .mask_detection()
                  .resize((2,2))
                  .binarize()
                  .ocr("")
                .end_branch()
              .end_split()
            .end_branch()
          
          .end_split()
          .build()
      )
    
    pipeline.run(img_bytes)


if __name__ == "__main__":
    main()