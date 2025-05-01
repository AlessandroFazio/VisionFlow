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
    builder = VisionFlow.pipeline_builder(config)
    
    pipeline = (
        builder.pipeline("name")
          .detect("poker/table_detection")
          .route_detection()
            .route("card")
              .filter(min_conf=0.70)
              .swap_image()
              .classify("poker/card_classifier")
            .end_route()
            .route("player_info")
              .filter(min_conf=0.70)
              .swap_image()
              .resize((2,2))
              .binarize()
              .ocr("poker/player_info_ocr")
            .end_route()
            .route("player_in")
              .filter(min_conf=0.70)
            .end_route()
            .route("dealer_button")
              .filter(min_conf=0.70)
            .end_route()
            .route("chips_amount")
              .filter(min_conf=0.70)
              .swap_image()
              .detect("poker/chips_detector")
              .route_detection()
                .route("chips")
                  .filter(min_conf=0.70)
                  .swap_image(exclude_detection=True)
                  .resize((2,2))
                  .binarize()
                  .ocr("")
                .end_route()
              .end_detection_router()
            .end_route()
          .end_detection_router()
          .build()
      )
    
    pipeline.run(img_bytes)


if __name__ == "__main__":
    main()