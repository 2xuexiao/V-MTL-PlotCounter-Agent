import cv2
import numpy as np
import os
from pathlib import Path
import torch
import sys
import subprocess
import tempfile
import shutil
import matplotlib.pyplot as plt
from matplotlib.cm import jet as cmap

# Add the tasselnetv2plus-master-yolo path to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../tasselnetv2plus-master-yolo')))

from hlnet_v2pp_variety_mtl6 import CountingModels, Normalizer
from utils import recover_countmap


class TasselNetModelMTL:

    def __init__(self, model_path):

        self.model_path = model_path

        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Initialize model structure
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(f"Creating CountingModels instance, using device: {self.device}")
        self.model = CountingModels(arc='tasselnetv2plus', input_size=64, output_stride=8)

        # Load model weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Create new state dict, removing 'module.' prefix
            new_state_dict = {}
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Process keys with "module." prefix
            for k, v in state_dict.items():
                name = k
                if name.startswith('module.'):
                    name = name[7:]  # Remove 'module.' prefix
                new_state_dict[name] = v

            # Load processed state dict
            self.model.load_state_dict(new_state_dict)
            print(f"Successfully loaded MTL model state dict, processed module prefix")

            # Set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            print(f"MTL model loaded to device: {self.device}")
        except Exception as e:
            print(f"Error loading MTL model: {str(e)}")
            raise RuntimeError(f"MTL model loading failed: {str(e)}")

        self.input_size = 64  #
        self.output_stride = 8
        self.resize_ratio = 0.5
        self.img_mean = [0.23324092, 0.22439253, 0.20940149]
        self.img_std = [0.17052431, 0.1618571, 0.15176316]

        print(f"MTL model loaded: {model_path}")

    def preprocess_image(self, image):

        print(f"MTL original preprocessed image size: {image.shape}")

        h, w = image.shape[:2]
        new_h = int(np.ceil(h * self.resize_ratio))  # Use np.ceil for rounding up
        new_w = int(np.ceil(w * self.resize_ratio))  # Use np.ceil for rounding up

        print(f"Image size after applying resize_ratio={self.resize_ratio}: {new_w}x{new_h}")
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        imh, imw = image.shape[0], image.shape[1]
        print(f"MTL final preprocessed image size: {imw}x{imh}")

        image = image.astype(np.float32) / 255.0

        for i in range(3):
            image[:, :, i] = (image[:, :, i] - self.img_mean[i]) / self.img_std[i]

        image = image.transpose((2, 0, 1))

        c, h, w = image.shape
        pad_h = 0 if h % self.output_stride == 0 else self.output_stride - (h % self.output_stride)
        pad_w = 0 if w % self.output_stride == 0 else self.output_stride - (w % self.output_stride)

        padded_h, padded_w = h, w

        if pad_h > 0 or pad_w > 0:
            print(f"MTL applying ZeroPadding: height+{pad_h}, width+{pad_w}")
            padded_image = np.zeros((c, h + pad_h, w + pad_w), dtype=np.float32)
            padded_image[:, :h, :w] = image
            image = padded_image
            padded_h, padded_w = h + pad_h, w + pad_w
            print(f"MTL image size after padding: {padded_w}x{padded_h}")

        image_tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)

        return image_tensor, padded_h, padded_w

    def predict(self, image):
        try:
            original_height, original_width = image.shape[:2]
            print(f"MTL original image size: {original_width}x{original_height}")

            print("Starting MTL model image preprocessing...")
            input_tensor, padded_h, padded_w = self.preprocess_image(image)

            print(f"MTL input tensor shape: {input_tensor.shape}, device: {input_tensor.device}")

            print("Starting MTL model inference...")
            with torch.no_grad():
                try:
                    model_output = self.model(input_tensor, is_normalize=False)

                    class_info = None
                    if isinstance(model_output, tuple):
                        output_save = model_output[0]
                        class_info = model_output[1]
                        print(f"MTL model returned tuple output, using first element shape: {output_save.shape}")
                        print(f"Class info shape: {class_info.shape}")

                        if isinstance(class_info, torch.Tensor):
                            class_probs = torch.nn.functional.softmax(class_info, dim=1)
                            _, predicted_class = torch.max(class_probs, 1)
                            class_info = {
                                'class_index': predicted_class.item(),
                                'probabilities': class_probs.squeeze().cpu().numpy().tolist()
                            }
                            print(
                                f"Predicted class: {class_info['class_index']}, probabilities: {class_info['probabilities']}")
                    else:
                        output_save = model_output
                        print(f"MTL model returned single output shape: {output_save.shape}")

                    # Normalize using Normalizer.gpu_normalizer
                    density_map = Normalizer.gpu_normalizer(output_save, padded_h, padded_w, self.input_size,
                                                            self.output_stride)

                    # Ensure numpy array
                    if isinstance(density_map, torch.Tensor):
                        density_map = density_map.cpu().numpy()

                    print(f"MTL normalized density map shape: {density_map.shape}")
                except Exception as model_error:
                    print(f"MTL model inference error: {str(model_error)}")
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(f"MTL model inference failed: {str(model_error)}")

            density_map = np.clip(density_map, 0, None)

            count = float(np.sum(density_map))
            print(f"MTL estimated count: {count}")

            # Directly use recover_countmap function for visualization output
            if isinstance(output_save, torch.Tensor):
                output_save = output_save.squeeze().cpu().numpy()

            output_save = np.clip(output_save, 0, None)

            # Use recover_countmap to restore density map to original size
            try:
                # Create actual numpy array (not dict) with shape [batch, channels, height, width]
                fake_tensor = np.zeros((1, 3, padded_h, padded_w), dtype=np.float32)
                output_viz = recover_countmap(output_save, fake_tensor, self.input_size, self.output_stride)

                # Manually resize if returned size doesn't match expected
                if output_viz.shape[:2] != (original_height, original_width):
                    output_viz = cv2.resize(output_viz, (original_width, original_height),
                                            interpolation=cv2.INTER_CUBIC)
            except Exception as e:
                print(f"MTL recover_countmap error, using simple resize as fallback: {str(e)}")
                # Fallback to simple resize if recover_countmap fails
                output_viz = cv2.resize(density_map, (original_width, original_height), interpolation=cv2.INTER_CUBIC)

            print("MTL prediction completed")
            return output_viz, count, class_info

        except Exception as e:
            print(f"Error during MTL prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"MTL model prediction failed: {str(e)}")

    def process_image(self, image_path):
        try:
            from pathlib import Path

            print(f"MTL reading image: {image_path}")
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to read image: {image_path}")

            height, width = img.shape[:2]
            print(f"MTL original image size: {width}x{height}")

            print("Starting MTL model image prediction...")
            density_map, count, class_info = self.predict(img)

            print("MTL prediction completed, generating visualization results...")

            density_map = density_map / (density_map.max() + 1e-12)
            try:
                colored_density = cmap(density_map) * 255.0
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                nh, nw = colored_density.shape[:2]
                rgb_img = cv2.resize(rgb_img, (nw, nh), interpolation=cv2.INTER_CUBIC)

                output_save = 0.5 * rgb_img + 0.5 * colored_density[:, :, 0:3]
                result = cv2.cvtColor(output_save.astype(np.uint8), cv2.COLOR_RGB2BGR)
            except Exception as viz_error:
                print(f"MTL visualization error: {str(viz_error)}")
                density_map_normalized = (density_map * 255).astype(np.uint8)
                density_map_color = cv2.applyColorMap(density_map_normalized, cv2.COLORMAP_JET)
                result = cv2.addWeighted(img, 0.5, density_map_color, 0.5, 0)

            result_filename = f"result_mtl_{Path(image_path).name}"

            base_dir = Path(__file__).resolve().parent.parent.parent
            results_dir = base_dir / "static" / "results"
            results_dir.mkdir(exist_ok=True, parents=True)

            result_path = results_dir / result_filename
            print(f"MTL saving result image to: {result_path}")

            os.makedirs(os.path.dirname(result_path), exist_ok=True)

            print("MTL saving result image...")
            success = cv2.imwrite(str(result_path), result)
            if not success:
                print(f"MTL failed to save image: {result_path}")
                # Try saving with different format
                print("Attempting to save as PNG...")
                png_path = str(result_path).replace(result_path.suffix, '.png')
                success = cv2.imwrite(png_path, result)
                if success:
                    print(f"MTL successfully saved as PNG: {png_path}")
                    result_filename = result_filename.replace(result_path.suffix, '.png')
                else:
                    print("MTL all saving attempts failed")
            else:
                print(f"MTL successfully saved image to: {result_path}")

            avg_density_value = float(density_map.mean())
            density_per_unit = float(count / (height * width / 10000))  # Crop count per 10k pixels

            # Define crop class mapping
            crop_classes = {
                0: "东豆 1号",  # Dongdou 1
                1: "铁丰 31号"  # Tiefeng 31
            }

            # English crop class mapping
            crop_classes_en = {
                0: "Dongdou 1",
                1: "Tiefeng 31"
            }

            # Get crop type name
            crop_type = "未知"
            crop_type_en = "Unknown"
            if class_info and 'class_index' in class_info:
                crop_type = crop_classes.get(class_info['class_index'], "未知")
                crop_type_en = crop_classes_en.get(class_info['class_index'], "Unknown")

            # Return analysis results
            print("MTL returning analysis results...")
            return {
                "count": int(round(count)),
                "avg_density_value": avg_density_value,
                "density_per_unit": density_per_unit,
                "image_dimensions": f"{width}x{height}",
                "result_image": f"/static/results/{result_filename}",
                "model_type": "MTL模型",
                "crop_type": crop_type,
                "crop_type_en": crop_type_en,
                "class_info": class_info
            }
        except Exception as e:
            print(f"Error processing image with MTL: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"MTL image processing failed: {str(e)}")


async def process_image_with_mtl_model(image_path, model_path=None):
    try:
        print(f"\nProcessing image with MTL model: {image_path}")

        # Set model path
        if model_path is None:
            # Default to using model in application directory
            base_dir = Path(os.path.dirname(__file__)).parent.parent.parent
            model_path = str(
                base_dir / "tasselnetv2plus-master-yolo" / "snapshots" / "wec" / "tasselnetv2plus-soybeanPlot-20-0.5-512-8-v2pp_variety_mtl6_0.01" / "model_best.pth.tar")

        # Create model instance
        model = TasselNetModelMTL(model_path)

        # Process image
        result = model.process_image(image_path)
        return result
    except Exception as e:
        print(f"Error processing image with MTL model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise Exception(f"MTL model processing failed: {str(e)}")
