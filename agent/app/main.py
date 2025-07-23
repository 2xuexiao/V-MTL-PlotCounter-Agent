from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import cv2
import numpy as np
import uuid
import aiofiles
from datetime import datetime
import json
import base64
import httpx
import os
import io
import asyncio
from PIL import Image
import re
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from .models.tasselnet.model_inference import process_image_with_tasselnet, process_image_with_original_model
from .models.tasselnet.model_inference_mtl import process_image_with_mtl_model
import shutil

app = FastAPI()

# Set static files and templates
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Create upload and result directories
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULTS_DIR = BASE_DIR / "static" / "results"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

latest_analysis = None

# Ollama API
OLLAMA_API_BASE = "http://127.0.0.1:11434/api"
GEMMA_MODEL = "gemma3:4b"  # Gemma 3B-4b

async def query_ollama_chat(messages, image_path=None, system_prompt=None):

    try:
        payload = {
            "model": GEMMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 8192,
                "repeat_penalty": 1.1
            }
        }

        if system_prompt:
            payload["system"] = system_prompt

        if image_path and os.path.exists(image_path) and messages and messages[-1]["role"] == "user":
            try:
                image = Image.open(image_path)
                max_size = (1024, 1024)
                image.thumbnail(max_size, Image.Resampling.LANCZOS)

                buffered = io.BytesIO()
                image.save(buffered, format="JPEG", quality=85)
                img_data = buffered.getvalue()

                base64_img = base64.b64encode(img_data).decode("utf-8")

                if "images" not in messages[-1]:
                    messages[-1]["images"] = []
                messages[-1]["images"].append(base64_img)
                print(f"Image added to the message: {Path(image_path).name}, Size: {len(img_data)/1024:.1f} KB")
            except Exception as e:
                print(f"Error adding image to message: {str(e)}")

        print(f"Sending request to {OLLAMA_API_BASE}/chat")
        print(f"Model: {GEMMA_MODEL}")
        # payload information
        payload_size = len(str(payload))
        print(f"Request size: {payload_size/1024:.1f} KB")
        print(f"Number of messages: {len(messages)}")
        if messages and len(messages) > 0:
            print(f"Last message: {messages[-1].get('content', '')[:100]}..." if len(messages[-1].get('content', '')) > 100 else f"最后一条消息: {messages[-1].get('content', '')}")

        
        # Call Ollama API
        max_retries = 2
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    print(f"Send request to Ollama API(Attempt{retry_count+1})")
                    response = await client.post(f"{OLLAMA_API_BASE}/chat", json=payload)
                    print(f"Received response status code: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        message = result.get("message", {})
                        model_response = message.get("content", "")
                        print(f"Successfully obtained model response, length: {len(model_response)}")
                        return model_response
                    elif response.status_code == 503 and retry_count < max_retries:
                        error_text = response.text
                        print(f"Ollama API returns a 503 error, waiting 5 seconds before retrying...")
                        last_error = Exception(f"Ollama API error: Status code {response.status_code}")
                        await asyncio.sleep(5)
                    else:
                        error_text = response.text
                        print(f"Ollama API error: Status code {response.status_code}, Response: {error_text[:200]}")
                        raise Exception(f"Ollama API error: Status code {response.status_code}")
            
            except httpx.TimeoutException:
                print(f"Ollama API request timeout (Attempt{retry_count+1})")
                last_error = Exception("Request timeout, model processing time is too long")
                if retry_count < max_retries:
                    await asyncio.sleep(3)
                else:
                    raise last_error
            
            except httpx.RequestError as e:
                print(f"Ollama API request error: {str(e)}")
                raise Exception(f"Request error: {str(e)}")
            
            retry_count += 1

        if last_error:
            raise last_error
    
    except Exception as e:
        print(f"Error calling Ollama API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model call failed: {str(e)}")

async def query_ollama_multimodal(prompt, image_path=None, system_prompt=None):

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    return await query_ollama_chat(messages, image_path, system_prompt)

async def process_image(image_path):
    try:
        print("\n" + "="*50)
        print(f"Start processing image: {image_path}")

        model_path = BASE_DIR.parent / "tasselnetv2plus-master-yolo" / "snapshots" / "wec" / "tasselnetv2plus-soybeanPlot-20-0.5-512-8-v2pp" / "model_best.pth.tar"
        print(f"Use model path: {model_path}")

        import sys, os
        sys.path.append(str(BASE_DIR.parent / "tasselnetv2plus-master-yolo"))
        
        import torch
        import cv2
        import numpy as np
        from hlnet_v2pp_variety_mtl6 import CountingModels

        img = cv2.imread(str(image_path))
        if img is None:
            raise HTTPException(status_code=400, detail="Cannot read image file")

        height, width = img.shape[:2]
        print(f"Original image size: {width}x{height}")

        new_width = width - (width % 8)
        new_height = height - (height % 8)
        if new_width != width or new_height != height:
            img = cv2.resize(img, (new_width, new_height))
            print(f"Resize image to multiples of 8: {new_width}x{new_height}")
            height, width = new_height, new_width
        
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = CountingModels(arc='tasselnetv2plus', input_size=64, output_stride=8)
        
        # Load weights
        checkpoint = torch.load(str(model_path), map_location=device)

        new_state_dict = {}
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'):
                name = name[7:]
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        print(f"Model loaded")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_mean = [0.23324092, 0.22439253, 0.20940149]
        img_std = [0.17052431, 0.1618571, 0.15176316]
        
        img_float = img_rgb.astype(np.float32) / 255.0
        for i in range(3):
            img_float[:, :, i] = (img_float[:, :, i] - img_mean[i]) / img_std[i]

        img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0).to(device)

        print("Perform model inference...")
        with torch.no_grad():
            density_map = model(img_tensor)
            if isinstance(density_map, torch.Tensor):
                density_map = density_map.squeeze().cpu().numpy()

        print("Calculation result...")
        count = float(np.sum(density_map))
        print(f"Estimated count: {count}")

        density_map_resized = cv2.resize(density_map, (width, height))
        if np.max(density_map_resized) > 0:
            density_map_normalized = (density_map_resized * 255 / np.max(density_map_resized)).astype(np.uint8)
        else:
            density_map_normalized = np.zeros((height, width), dtype=np.uint8)

        density_map_color = cv2.applyColorMap(density_map_normalized, cv2.COLORMAP_JET)

        result = cv2.addWeighted(img, 0.6, density_map_color, 0.4, 0)

        from pathlib import Path
        result_filename = f"result_{Path(image_path).name}"
        result_path = RESULTS_DIR / result_filename
        cv2.imwrite(str(result_path), result)
        print(f"Result saved: {result_path}")

        avg_density_value = float(density_map_resized.mean())
        density_per_unit = float(count / (height * width / 10000))

        analysis_report = (
            f"基于TasselNetV2Plus模型的作物计数分析，我获取了以下数据：\n\n"
            f"1. 作物植株数量：检测到约{count}株作物\n"
            f"2. 种植密度指标：每万像素约有{density_per_unit:.2f}株作物\n"
            f"2. 图像尺寸：{width} x {height}\n\n"
            f"注意事项：\n"
            f"- 这些数据基于密度图估计，可能与实际数量有一定误差\n"
            f"- 影响计数准确性的因素包括图像质量、光照条件等\n"
            f"- 您可以在聊天中提供更多信息（如生长阶段、种植密度等），以获得更具针对性的分析"
        )

        result_data = {
            "count": count,
            "avg_density_value": avg_density_value,
            "density_per_unit": density_per_unit, 
            "image_dimensions": f"{width}x{height}",
            "result_image": f"/static/results/{result_filename}",
            "analysis_report": analysis_report
        }
        
        print("Processing completed")
        return result_data
        
    except Exception as e:
        print(f"Error occurred during model processing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model processing failed: {str(e)}")

async def check_ollama_api():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            print(f"Checking Ollama API availability...")
            response = await client.get(f"{OLLAMA_API_BASE}/tags")
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                model_names = [m.get("name") for m in models]
                print(f"Ollama API available, loaded models: {', '.join(model_names)}")
                model_exists = any(GEMMA_MODEL in m.get("name", "") for m in models)
                if model_exists:
                    print(f"✅ Model found: {GEMMA_MODEL}")
                else:
                    print(f"⚠️ Warning: Model {GEMMA_MODEL} not found")
                return True, model_names
            else:
                print(f"Ollama API unavailable, status code: {response.status_code}")
                return False, None
    except Exception as e:
        print(f"Error checking Ollama API: {str(e)}")
        return False, None

@app.on_event("startup")
async def startup_event():
    api_available, models = await check_ollama_api()
    if api_available:
        if isinstance(models, list) and models:
            print(f"Ollama service available, models: {', '.join(models)}")
            model_exists = any(GEMMA_MODEL in m for m in models)
            if not model_exists:
                print(f"⚠️ Warning: Model {GEMMA_MODEL} not found, please download with 'ollama pull {GEMMA_MODEL}'")
                print(f"To use different model, modify GEMMA_MODEL in main.py")
        else:
            print(f"Ollama service available but no loaded models found")
    else:
        print(f"⚠️ Warning: Ollama service unavailable, chat will use fallback responses")
        print(f"Please ensure Ollama service is running: ollama serve")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    try:
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        upload_path = UPLOAD_DIR / unique_filename

        async with aiofiles.open(upload_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        print(f"Image saved to: {upload_path}")

        result = await process_image_with_mtl_model(str(upload_path))

        result_image_url = result.get('result_image')
        count = result.get('count', 0)
        density_per_unit = result.get('density_per_unit')
        image_dimensions = result.get('image_dimensions')
        crop_type_zh = result.get('crop_type', '未知')
        crop_type_en = result.get('crop_type_en', 'Unknown')

        global latest_analysis
        latest_analysis = {
            "image_path": str(upload_path),
            "result_image_path": result_image_url,
            "count": count,
            "density_per_unit": density_per_unit,
            "image_dimensions": image_dimensions,
            "crop_type": crop_type_zh,
            "crop_type_en": crop_type_en,
            "timestamp": datetime.now().isoformat()
        }

        return {
            "result_image": result_image_url,
            "density_map_path": result_image_url,
            "count": count,
            "density_per_unit": density_per_unit,
            "image_dimensions": image_dimensions,
            "crop_type": crop_type_zh,
            "crop_type_en": crop_type_en,
        }
            
    except Exception as e:
        print(f"Error occurred while processing uploaded file: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: Request):
    global latest_analysis
    try:
        data = await request.json()
        message = data.get("message", "")
        history = data.get("history", [])
        language = data.get("language", "zh")

        analysis = latest_analysis or {
            "count": 0,
            "avg_density_value": 0,
            "density_per_unit": 0,
            "image_dimensions": "未知" if language == "zh" else "unknown",
            "analysis_report": "暂无分析数据，请先上传图片进行分析。" if language == "zh" else "No analysis data available. Please upload an image for analysis."
        }

        latest_image = None
        original_image = None
        if latest_analysis and "result_image" in latest_analysis:
            result_image = latest_analysis["result_image"]
            if result_image.startswith("/static/results/result_"):
                original_filename = Path(result_image).name.replace("result_", "", 1)
                possible_original = UPLOAD_DIR / original_filename
                if possible_original.exists():
                    latest_image = str(possible_original)
                    original_image = original_filename
        
        # Attempt to use Ollama model
        ollama_available = False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{OLLAMA_API_BASE}/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m.get("name", "") for m in models]

                    if any(GEMMA_MODEL in name for name in model_names):
                        ollama_available = True
                        print(f"Ollama API available, model found: {GEMMA_MODEL}")
                    else:
                        print(f"Ollama API available, but model not found: {GEMMA_MODEL}")
                else:
                    print(f"Ollama API unavailable: {response.status_code}")
        except Exception as e:
            print(f"Ollama API connection failed: {str(e)}")

        # If Ollama is available, use the model to generate a response
        if ollama_available:
            try:
                if language == "en":
                    system_prompt = (
                        "You are a professional agricultural analysis AI advisor with extensive agronomic knowledge and data analysis capabilities, specializing in crop density analysis, planting optimization, and field management.\n\n"
                        "Professional Guidelines:\n"
                        "1. Data-based Analysis: Combine image analysis data with user-provided field information (crop types, soil conditions, climate environment, etc.) for comprehensive evaluation\n"
                        "2. Differentiated Analysis: Apply appropriate optimal density standards and planting parameters for different crop types\n"
                        "3. Scientific Rigor: Clearly distinguish between observable data and inferred conclusions, avoiding excessive inference about information not presented in the image\n"
                        "4. Practical Orientation: Provide specific, feasible agricultural management measures and intervention plans, rather than general advice\n"
                        "5. Comprehensive Consideration: When evaluating density, consider crop characteristics, growth stage requirements, soil fertility status, water availability, etc.\n"
                        "6. Structured Output: Present analysis results in clear sections and points for easy understanding and implementation\n\n"
                        
                        f"Current Image Analysis Basic Data:\n"
                        f"- Predicted seeding count: {analysis['count']} plants\n"
                        f"- crop_type: {analysis.get('crop_type', 'Unknown')}\n"
                        f"- Density Metric: Approximately {analysis.get('density_per_unit', 0):.2f} plants per 10,000 pixels\n"
                        f"- Image Dimensions: {analysis.get('image_dimensions', 'unknown')}\n"
                        f"- Image File: {original_image or 'unspecified'}\n\n"
                        
                        # "Analysis Framework and Response Structure:\n"
                        # "1. Overview: Briefly confirm the issue being analyzed and understand the information provided by the user\n"
                        # "2. Density Assessment: Evaluate whether the current density is appropriate, considering crop type and growth stage\n"
                        # "3. Specific Analysis: Provide detailed analysis for user concerns (such as yield impact, pest & disease risk, irrigation efficiency, etc.)\n"
                        # "4. Improvement Suggestions: Provide 3-5 specific actionable improvement suggestions, including short-term and long-term measures\n"
                        # "5. Additional Information Needs: If necessary, suggest what additional key information the user should provide for more accurate analysis"
                    )

                    field_labels = {
                        "crop_type": "Crop Type",
                        "soil_type": "Soil Type",
                        "climate": "Climate Conditions",
                        "region": "Growing Region",
                        "growth_stage": "Growth Stage",
                        "irrigation": "Irrigation Method",
                        "planting_method": "Planting Method",
                        "field_size": "Field Size",
                        "season": "Current Season"
                    }

                    focus_areas = {
                        "pest": "When analyzing pest and disease risks, consider the relationship between current density and disease transmission risk, focus on early warning indicators, and provide comprehensive prevention and control measures.",
                        "disease": "When analyzing pest and disease risks, consider the relationship between current density and disease transmission risk, focus on early warning indicators, and provide comprehensive prevention and control measures.",
                        "yield": "When analyzing yield potential, focus on evaluating the impact of current density on yield, considering photosynthetic efficiency, nutrient competition, and actual field management conditions.",
                        "irrigation": "When evaluating irrigation plans, combine density and crop water requirements to provide precise irrigation recommendations, analyzing water use efficiency and optimization directions.",
                        "soil": "When analyzing soil adaptability, focus on the match between soil type and crop growth requirements, as well as nutrient supply capacity and improvement directions at the current density."
                    }

                    prompt_prefix = f"Based on the image analysis report:\n{analysis.get('analysis_report', '')}\n\nUser question: {message}"
                else:

                    system_prompt = (
                        "你是一位专业的农业分析AI顾问，拥有丰富的农学知识和数据分析能力，专长于作物密度分析、种植方案优化和农田管理。\n\n"
                        "专业指南：\n"
                        "1. 基于数据分析：将图像分析数据与用户提供的农田信息（作物种类、土壤条件、气候环境等）结合进行综合评估\n"
                        "2. 差异化分析：针对不同作物类型采用相应的最佳密度标准和种植参数进行分析\n"
                        "3. 科学严谨：明确区分可观测数据与推断结论，不对图像无法呈现的信息做过度推断\n"
                        "4. 实用导向：提供具体可行的农业管理措施和干预方案，而非泛泛而谈\n"
                        "5. 全面考量：评估密度时，综合考虑作物种类特性、生长阶段需求、土壤肥力状况、水分可得性等因素\n"
                        "6. 结构化输出：使用清晰的章节和要点呈现分析结果，便于用户理解和实施\n\n"
                        
                        f"当前图像分析基础数据：\n"
                        f"- 作物计数：{analysis['count']}株\n"
                        f"- 作物类型：{analysis.get('crop_type', '未知')}\n"
                        f"- 密度指标：每万像素约有{analysis.get('density_per_unit', 0):.2f}株作物\n"
                        f"- 图像尺寸：{analysis.get('image_dimensions', '未知')}\n"
                        f"- 图像文件：{original_image or '未指定'}\n\n"
                        
                        "分析框架与回答结构：\n"
                        "1. 概述：简明确认所分析的问题和理解用户提供的信息\n"
                        "2. 密度评估：结合作物种类和生长阶段，评估当前密度是否适宜\n"
                        "3. 具体分析：针对用户关注点提供详细分析（如产量影响、病虫害风险、灌溉效率等）\n"
                        "4. 改进建议：提供3-5点具体可操作的改进建议，包括短期和长期措施\n"
                        "5. 补充信息需求：如有必要，建议用户提供哪些额外关键信息以获得更精准分析"
                    )

                    field_labels = {
                        "crop_type": "作物类型",
                        "soil_type": "土壤类型",
                        "climate": "气候条件",
                        "region": "种植区域",
                        "growth_stage": "生长阶段",
                        "irrigation": "灌溉方式",
                        "planting_method": "种植方式",
                        "field_size": "田地规模",
                        "season": "当前季节"
                    }

                    focus_areas = {
                        "病虫害": "提供病虫害风险分析时，考虑当前密度与病虫害传播风险的关系，关注早期预警指标，提供预防和治理的综合措施。",
                        "产量": "分析产量潜力时，重点评估当前密度对产量的影响，考虑光合作用效率、养分竞争和实际田间管理条件。",
                        "灌溉": "评估灌溉方案时，结合密度和作物需水特性，提供精准灌溉建议，分析水分利用效率和优化方向。",
                        "土壤": "分析土壤适应性时，注重土壤类型与作物生长需求的匹配度，以及在当前密度下的养分供应能力和改良方向。"
                    }

                    prompt_prefix = f"基于图像分析报告：\n{analysis.get('analysis_report', '')}\n\n用户问题: {message}"

                extracted_info = {}

                extraction_patterns = {
                    "crop_type": [
                        r"种植的是\s*([^，。,；;]+)",
                        r"种植\s*([^，。,；;]+)",
                        r"我的\s*([^，。,；;]+)\s*农田",
                        r"这是\s*([^，。,；;]+)\s*的分布图",
                        r"种了\s*([^，。,；;]+)",
                        r"([^，。,；;]+)\s*田",
                        r"([^，。,；;]+)\s*地里",
                        # 英文模式
                        r"planted\s*([^,.;]+)",
                        r"growing\s*([^,.;]+)",
                        r"my\s*([^,.;]+)\s*field",
                        r"([^,.;]+)\s*crop"
                    ],
                    "soil_type": [
                        r"土壤[是为类型]+\s*([^，。,；;]+)",
                        r"土质[是为]+\s*([^，。,；;]+)",
                        r"([^，。,；;]+)\s*土壤",
                        r"地是\s*([^，。,；;]+)",
                        # 英文模式
                        r"soil[is type]+\s*([^,.;]+)",
                        r"([^,.;]+)\s*soil"
                    ],
                    "climate": [
                        r"气候[是为条件]+\s*([^，。,；;]+)",
                        r"天气[是为]+\s*([^，。,；;]+)",
                        r"环境[是为]+\s*([^，。,；;]+)",
                        r"天气条件[是为]+\s*([^，。,；;]+)",
                        # 英文模式
                        r"climate[is condition]+\s*([^,.;]+)",
                        r"weather[is]+\s*([^,.;]+)",
                        r"environment[is]+\s*([^,.;]+)"
                    ],
                    "region": [
                        r"位于\s*([^，。,；;]+)",
                        r"在\s*([^，。,；;]+)\s*地区",
                        r"([^，。,；;]+)\s*地区的",
                        r"来自\s*([^，。,；;]+)",
                        # 英文模式
                        r"located in\s*([^,.;]+)",
                        r"in\s*([^,.;]+)\s*region",
                        r"from\s*([^,.;]+)"
                    ],
                    "growth_stage": [
                        r"[生长阶段期]+[是为处于]+\s*([^，。,；;]+)",
                        r"现在是\s*([^，。,；;]+)\s*[阶段期]",
                        r"处于\s*([^，。,；;]+)\s*[阶段期]",
                        r"([^，。,；;]+)\s*生长期",
                        # 英文模式
                        r"growth stage[is]+\s*([^,.;]+)",
                        r"in\s*([^,.;]+)\s*stage",
                        r"during\s*([^,.;]+)\s*stage"
                    ],
                    "irrigation": [
                        r"灌溉[是采用使用方式]+\s*([^，。,；;]+)",
                        r"用\s*([^，。,；;]+)\s*灌溉",
                        r"([^，。,；;]+)\s*方式浇水",
                        # 英文模式
                        r"irrigation[is method]+\s*([^,.;]+)",
                        r"using\s*([^,.;]+)\s*irrigation",
                        r"watering with\s*([^,.;]+)"
                    ],
                    "planting_method": [
                        r"种植方式[是为采用]+\s*([^，。,；;]+)",
                        r"采用\s*([^，。,；;]+)\s*方式种植",
                        r"([^，。,；;]+)\s*栽培方式",
                        # 英文模式
                        r"planting method[is]+\s*([^,.;]+)",
                        r"using\s*([^,.;]+)\s*method",
                        r"([^,.;]+)\s*cultivation"
                    ],
                    "field_size": [
                        r"农田[大小面积是为有]+\s*([^，。,；;]+)",
                        r"([^，。,；;]+)\s*[亩公顷平方米]",
                        r"地[大小面积是为有]+\s*([^，。,；;]+)",
                        # 英文模式
                        r"field size[is]+\s*([^,.;]+)",
                        r"([^,.;]+)\s*acres",
                        r"([^,.;]+)\s*hectares"
                    ],
                    "season": [
                        r"季节[是为]+\s*([^，。,；;]+)",
                        r"现在是\s*([^，。,；;]+)\s*季",
                        r"([春夏秋冬雨旱]季)",
                        # 英文模式
                        r"season[is]+\s*([^,.;]+)",
                        r"during\s*([^,.;]+)\s*season",
                        r"in\s*(spring|summer|autumn|winter|fall)"
                    ]
                }

                for field, patterns in extraction_patterns.items():
                    for pattern in patterns:
                        match = re.search(pattern, message, re.IGNORECASE)
                        if match:
                            extracted_text = match.group(1).strip()
                            if extracted_text and len(extracted_text) < 20:
                                extracted_info[field] = extracted_text
                                break

                if extracted_info:
                    additional_info = []

                    for field, value in extracted_info.items():
                        if field in field_labels:
                            additional_info.append(f"- {field_labels[field]}：{value}")
                    
                    if additional_info:
                        if language == "en":
                            system_prompt += f"\n\nUser-provided field information:\n{chr(10).join(additional_info)}"
                            print(f"Extracted {len(additional_info)} key information items from user message")
                        else:
                            system_prompt += f"\n\n用户提供的农田信息：\n{chr(10).join(additional_info)}"
                            print(f"Extracted {len(additional_info)} key pieces of information from user message")

                has_keyword = False
                for keyword, guidance in focus_areas.items():
                    if keyword in message.lower():
                        if language == "en":
                            system_prompt += f"\n\nAnalysis Focus: {guidance}"
                        else:
                            system_prompt += f"\n\n专业分析重点：{guidance}"
                        has_keyword = True
                        break
                

                prompt = prompt_prefix

                if history:
                    if language == "en":
                        context = "\nConversation History:\n"
                        for entry in history[-3:]:
                            role = entry.get("role", "")
                            content = entry.get("content", "")
                            if role == "user":
                                context += f"User: {content}\n"
                            elif role == "assistant":
                                context += f"AI: {content}\n"
                    else:
                        context = "\n历史对话：\n"
                        for entry in history[-3:]:
                            role = entry.get("role", "")
                            content = entry.get("content", "")
                            if role == "user":
                                context += f"用户: {content}\n"
                            elif role == "assistant":
                                context += f"AI: {content}\n"
                    prompt = context + "\n" + prompt

                    print(f"\n===== Model Invocation Details =====")
                    print(f"Language setting: {language}")
                    print(f"Invoking Ollama model: {GEMMA_MODEL}")
                    print(f"Image path: {latest_image}")

                    if extracted_info:
                        print(f"\nExtracted information:")
                        for field, value in extracted_info.items():
                            print(f"  - {field}: {value}")

                    print(f"\nKeyword matching results:")
                    for keyword in focus_areas.keys():
                        if keyword in message.lower():
                            print(f"  - Matched keyword: {keyword}")

                    if history:
                        print(f"\nChat history: {len(history)} messages")

                    print(f"\nSystem prompt ({len(system_prompt)} chars): {system_prompt[:900]}...")
                    print(f"User prompt ({len(prompt)} chars): {prompt[:900]}...")

                    # Call multimodal model
                    if latest_image and os.path.exists(latest_image):
                        print(f"Making multimodal request with image: {latest_image}")
                        response = await query_ollama_multimodal(prompt, latest_image, system_prompt)
                    else:
                        print("Making text-only request")
                        response = await query_ollama_multimodal(prompt, None, system_prompt)

                    print(f"Received model response, length: {len(response)}")
                    return JSONResponse(content={"response": response})

            except Exception as e:
                print(f"Ollama model invocation error: {str(e)}")
                # Fallback to rule-based response when error occurs
                ollama_available = False

        # If Ollama is unavailable or encounters an error, use a default response based on objective data
        if not ollama_available:
            if language == "en":
                if any(keyword in message.lower() for keyword in ["density", "distribution"]):
                    response = f"According to the analysis, there are approximately {analysis.get('density_per_unit', 0):.2f} plants per 10,000 pixels in the image. To assess whether this density is appropriate, we need to know the specific crop type and growing environment."
                elif any(keyword in message.lower() for keyword in ["count", "number"]):
                    response = f"The image contains {analysis['count']} detected plants."
                elif any(keyword in message.lower() for keyword in ["growth", "condition"]):
                    response = "From the image, I can only obtain information about the number and distribution of crops, not accurately judge their growth condition. To assess growth condition, we typically need to observe leaf color, size, plant height, or have information about the crop variety and growth stage."
                elif any(keyword in message.lower() for keyword in ["suggest", "optimize", "recommendation"]):
                    response = "To provide planting optimization suggestions, I need to understand your crop type, current growth stage, soil conditions, and climate environment. You can provide this information in your message to receive more targeted advice."
                elif any(keyword in message.lower() for keyword in ["report", "complete analysis"]):
                    response = analysis['analysis_report']
                else:
                    response = "I understand your question. Based on the image analysis, we detected the number and distribution of crops. To get more valuable analysis, you can ask specific questions and provide more background information, such as crop type and growing environment. You can also request to view the complete analysis report."

                response += "\n\n(Note: This is a system default response. The Ollama model is not connected. Please check if the Ollama service is running properly.)"
            else:
                if "密度" in message or "分布" in message:
                    response = f"根据分析，图像中每万像素约有{analysis.get('density_per_unit', 0):.2f}株作物。要评估这个密度是否合适，还需要了解具体的作物类型和生长环境。"
                elif "数量" in message or "计数" in message:
                    response = f"图像中共检测到{analysis['count']}株作物。"
                elif "生长" in message or "状况" in message:
                    response = "从图像中我只能获取作物的数量和分布情况，无法准确判断生长状况。要评估生长状况，通常需要观察叶片颜色、大小、植株高度等因素，或者提供作物的品种和生长阶段等信息。"
                elif "建议" in message or "优化" in message:
                    response = "要提供种植优化建议，我需要了解您种植的作物类型、当前生长阶段、土壤条件和气候环境等信息。您可以在消息中提供这些信息，以获得更具针对性的建议。"
                elif "报告" in message or "完整分析" in message:
                    response = analysis['analysis_report']
                else:
                    response = "我理解您的问题。根据图像分析，我们检测到了作物的数量和分布情况。要获得更有价值的分析，您可以询问具体问题并提供更多背景信息，如作物类型、种植环境等。您也可以要求查看完整分析报告。"

                response += "\n\n(注: 这是系统默认回答，Ollama模型无法连接。请检查Ollama服务是否正常运行。)"
        
            return JSONResponse(content={"response": response})
        
    except Exception as e:
        print(f"Chat processing error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": "处理请求时出错", "response": "非常抱歉，处理您的请求时出现了错误。请重试或联系管理员。"}
        )

@app.post("/generate_prompt")
async def generate_prompt(request: Request):
    global latest_analysis
    
    count = latest_analysis.get("count", 0) if latest_analysis else 0

    data = await request.json()
    language = data.get("language", "zh")

    if language == "en":
        options = {
            "count": count,
            "has_analysis": latest_analysis is not None,
            "options": {
                "crop_types": [
                    "Rice", "Wheat", "Corn", "Soybean", "Cotton", "Canola", "Potato", "Sugarcane", 
                    "Peanut", "Tea", "Fruit Trees", "Vegetables", "Sunflower", "Tobacco"
                ],
                "soil_types": [
                    "Black Soil", "Red Soil", "Clay Soil", "Sandy Soil", "Loam Soil", "Alkaline Soil", "Acidic Soil", 
                    "Calcareous Soil", "Saline-Alkaline Soil", "Sandy Soil"
                ],
                "irrigation_methods": [
                    "Drip Irrigation", "Sprinkler Irrigation", "Micro Irrigation", "Furrow Irrigation", "Flood Irrigation",
                    "Spray Irrigation", "No Irrigation", "Rain-fed Irrigation", "Timed Irrigation", "Smart Irrigation"
                ],
                "growth_stages": [
                    "Seeding Stage", "Emergence Stage", "Seedling Stage", "Growth Stage", "Flowering Stage", "Fruiting Stage", 
                    "Maturity Stage", "Pre-harvest", "Wintering Stage"
                ],
                "seasons": [
                    "Spring", "Summer", "Autumn", "Winter", "Rainy Season", "Dry Season"
                ],
                "climate_conditions": [
                    "Warm and Humid", "Dry", "Semi-arid", "Tropical", "Subtropical", "Temperate", "Cold", 
                    "High Temperature", "High Rainfall", "Low Rainfall"
                ],
                "planting_methods": [
                    "Row Planting", "Mixed Planting", "Intercropping", "Vertical Planting", "Crop Rotation", "Terrace Planting", 
                    "Dense Planting", "Sparse Planting", "Field Cultivation", "Grid Planting"
                ],
                "field_sizes": [
                    "Less than 0.5 acre", "0.5-2.5 acres", "2.5-5 acres", "5-25 acres", "25-50 acres", "Over 50 acres"
                ]
            },
            "templates": [
                "种植密度分析",  # Keep original template key-values; frontend shows translations by language
                "产量优化建议",
                "灌溉方案建议",
                "病虫害风险评估",
                "土壤适应性分析",
                "种植方式优化",
                "季节性种植建议"
            ]
        }
    else:
        options = {
            "count": count,
            "has_analysis": latest_analysis is not None,
            "options": {
                "crop_types": [
                    "水稻", "小麦", "玉米", "大豆", "棉花", "油菜", "马铃薯", "甘蔗", 
                    "花生", "茶叶", "果树", "蔬菜", "向日葵", "烟草"
                ],
                "soil_types": [
                    "黑土", "红土", "粘土", "砂质土", "壤土", "碱性土", "酸性土", 
                    "石灰质土", "盐碱土", "沙土"
                ],
                "irrigation_methods": [
                    "滴灌", "喷灌", "微灌", "沟灌", "漫灌", "喷洒灌溉", "无灌溉", 
                    "雨水灌溉", "定时灌溉", "智能灌溉"
                ],
                "growth_stages": [
                    "播种期", "出苗期", "幼苗期", "生长期", "开花期", "结果期", 
                    "成熟期", "收获前", "越冬期"
                ],
                "seasons": [
                    "春季", "夏季", "秋季", "冬季", "雨季", "旱季"
                ],
                "climate_conditions": [
                    "温暖湿润", "干燥", "半干旱", "热带", "亚热带", "温带", "寒冷", 
                    "高温", "多雨", "少雨"
                ],
                "planting_methods": [
                    "行间种植", "混合种植", "间作套种", "立体种植", "轮作", "梯田种植", 
                    "密植", "稀植", "大田栽培", "网格种植"
                ],
                "field_sizes": [
                    "小于1亩", "1-5亩", "5-10亩", "10-50亩", "50-100亩", "100亩以上"
                ]
            },
            "templates": [
                "种植密度分析",
                "产量优化建议",
                "灌溉方案建议",
                "病虫害风险评估",
                "土壤适应性分析",
                "种植方式优化",
                "季节性种植建议"
            ]
        }
    
    return JSONResponse(content=options)

# Add a new endpoint to process user selections and generate the final prompt
class PromptOptions(BaseModel):
    templates: List[str]
    options: Dict[str, Any]
    language: Optional[str] = "zh"
    
@app.post("/generate_prompt_from_options")
async def generate_prompt_from_options(options: PromptOptions):
    global latest_analysis
    
    count = latest_analysis.get("count", 0) if latest_analysis else 0
    selected_templates = options.templates
    user_options = options.options
    language = options.language
    
    # Extract various options from user selections
    crop_type = user_options.get("crop_type", "")
    soil_type = user_options.get("soil_type", "")
    irrigation = user_options.get("irrigation", "")
    growth_stage = user_options.get("growth_stage", "")
    season = user_options.get("season", "")
    climate = user_options.get("climate", "")
    planting_method = user_options.get("planting_method", "")
    field_size = user_options.get("field_size", "")

    final_prompt = ""

    template_prompts = []

    def format_option(value, prefix="", suffix=""):
        if value and value.strip():
            return f"{prefix}{value}{suffix}"
        return ""
    
    # Select label text according to language settings
    if language == "en":
        # English label
        crop_label = "Crop Type: "
        soil_label = "Soil Type: "
        climate_label = "Climate Conditions: "
        irrigation_label = "Irrigation: "
        growth_label = "Growth Stage: "
        season_label = "Season: "
        planting_label = "Planting Method: "
        field_label = "Field Size: "
        
        # English template title mapping
        template_titles = {
            "种植密度分析": "Planting Density Analysis",
            "产量优化建议": "Yield Optimization Suggestions",
            "灌溉方案建议": "Irrigation Plan Recommendations",
            "病虫害风险评估": "Pest & Disease Risk Assessment",
            "土壤适应性分析": "Soil Adaptability Analysis", 
            "种植方式优化": "Planting Method Optimization",
            "季节性种植建议": "Seasonal Planting Recommendations"
        }
    else:
        # Chinese label
        crop_label = "作物类型："
        soil_label = "土壤类型："
        climate_label = "气候条件："
        irrigation_label = "灌溉方式："
        growth_label = "生长阶段："
        season_label = "季节："
        planting_label = "种植方式："
        field_label = "田地规模："
        
        # Chinese templates use the original title
        template_titles = {
            "种植密度分析": "种植密度分析",
            "产量优化建议": "产量优化建议",
            "灌溉方案建议": "灌溉方案建议",
            "病虫害风险评估": "病虫害风险评估",
            "土壤适应性分析": "土壤适应性分析", 
            "种植方式优化": "种植方式优化",
            "季节性种植建议": "季节性种植建议"
        }
    
    # basic information
    crop_info = format_option(crop_type, crop_label)
    soil_info = format_option(soil_type, soil_label)
    climate_info = format_option(climate, climate_label)
    irrigation_info = format_option(irrigation, irrigation_label)
    growth_info = format_option(growth_stage, growth_label)
    season_info = format_option(season, season_label)
    planting_info = format_option(planting_method, planting_label)
    field_info = format_option(field_size, field_label)
    
    #
    base_info = [info for info in [crop_info, soil_info, climate_info, irrigation_info, 
                                   growth_info, season_info, planting_info, field_info] if info]
    
    #
    for template in selected_templates:
        template_title = template_titles.get(template, template)
        
        if language == "en":
            # English template
            if template == "种植密度分析":
                prompt = f"【Planting Density Analysis】In the image, {count} plants were detected. Please analyze whether this planting density is reasonable."
                if base_info:
                    prompt += f"\n\nMy field conditions:\n• " + "\n• ".join(base_info)
                prompt += "\n\nPlease focus on analyzing:\n1. Comparison of current density with standard density\n2. Potential impact of this density on crop growth\n3. Whether density adjustment is needed, and if so, how to proceed"
                template_prompts.append(prompt)
            
            elif template == "产量优化建议":
                prompt = f"【Yield Optimization Suggestions】Based on the analysis of {count} plants in the image, please provide yield optimization suggestions."
                if base_info:
                    prompt += f"\n\nMy field conditions:\n• " + "\n• ".join(base_info)
                prompt += "\n\nPlease provide:\n1. Potential impact of current planting on yield\n2. Key measures to increase yield (at least 3 points)\n3. Seasonal management tips"
                template_prompts.append(prompt)
            
            elif template == "灌溉方案建议":
                prompt = f"【Irrigation Plan Recommendations】Based on the count of {count} plants in the image analysis, please evaluate the current irrigation plan and provide optimization suggestions."
                if base_info:
                    prompt += f"\n\nCurrent field conditions:\n• " + "\n• ".join(base_info)
                prompt += "\n\nPlease analyze:\n1. Whether the current irrigation method is suitable for this density and crop type\n2. How to optimize the irrigation plan (frequency, water volume, etc.)\n3. Are there more efficient irrigation technologies to recommend"
                template_prompts.append(prompt)
            
            elif template == "病虫害风险评估":
                prompt = f"【Pest & Disease Risk Assessment】Based on the analysis of {count} plants in the image, please assess the potential pest and disease risks."
                if base_info:
                    prompt += f"\n\nField conditions:\n• " + "\n• ".join(base_info)
                prompt += "\n\nPlease analyze:\n1. Most likely types of pests and diseases under current planting density and environment\n2. Early warning signs for pests and diseases\n3. Prevention and control measures\n4. Whether pesticide use is recommended, and if so, what types should be chosen"
                template_prompts.append(prompt)
            
            elif template == "土壤适应性分析":
                prompt = f"【Soil Adaptability Analysis】Based on the analysis of {count} plants in the image, please evaluate the adaptability of current soil conditions."
                if base_info:
                    prompt += f"\n\nField conditions:\n• " + "\n• ".join(base_info)
                prompt += "\n\nPlease analyze in detail:\n1. Whether the current soil type is suitable for growing this crop\n2. Observed soil problems or advantages\n3. Specific soil improvement suggestions (if needed)\n4. Fertilizer choices suitable for the current soil"
                template_prompts.append(prompt)
            
            elif template == "种植方式优化":
                prompt = f"【Planting Method Optimization】Based on the distribution of {count} plants in the image, please analyze the current planting method and provide optimization suggestions."
                if base_info:
                    prompt += f"\n\nField conditions:\n• " + "\n• ".join(base_info)
                prompt += "\n\nPlease analyze:\n1. Pros and cons of the current planting method\n2. More suitable planting methods for this crop\n3. How to adjust row spacing and plant spacing to optimize growth space\n4. Specific implementation steps for improving the planting method"
                template_prompts.append(prompt)
            
            elif template == "季节性种植建议":
                prompt = f"【Seasonal Planting Recommendations】Based on the analysis of {count} plants in the image, please provide planting management suggestions for the current season."
                if base_info:
                    prompt += f"\n\nField conditions:\n• " + "\n• ".join(base_info)
                prompt += "\n\nPlease provide:\n1. Management points that need special attention in the current season\n2. Measures to cope with seasonal climate changes\n3. Preparation work for the next stage\n4. Methods to optimize environmental factors such as light and temperature"
                template_prompts.append(prompt)
        else:
            # Chinese template
            if template == "种植密度分析":
                prompt = f"【种植密度分析】图像中检测到{count}株作物，请分析这个种植密度是否合理。"
                if base_info:
                    prompt += f"\n\n我的农田情况：\n• " + "\n• ".join(base_info)
                prompt += "\n\n请重点分析：\n1. 当前密度与标准密度的比较\n2. 这种密度对作物生长可能的影响\n3. 是否需要调整密度，如需调整应如何操作"
                template_prompts.append(prompt)
            
            elif template == "产量优化建议":
                prompt = f"【产量优化建议】基于图像分析的{count}株作物计数结果，请提供产量优化建议。"
                if base_info:
                    prompt += f"\n\n我的农田情况：\n• " + "\n• ".join(base_info)
                prompt += "\n\n请提供：\n1. 当前种植情况对产量的潜在影响\n2. 提高产量的关键措施（至少3点）\n3. 季节性管理要点"
                template_prompts.append(prompt)
            
            elif template == "灌溉方案建议":
                prompt = f"【灌溉方案建议】基于图像分析计数的{count}株作物，请评估当前灌溉方案并提供优化建议。"
                if base_info:
                    prompt += f"\n\n当前农田情况：\n• " + "\n• ".join(base_info)
                prompt += "\n\n请分析：\n1. 当前灌溉方式是否适合这种密度和作物类型\n2. 如何优化灌溉计划（频率、水量等）\n3. 有没有更高效的灌溉技术推荐"
                template_prompts.append(prompt)
            
            elif template == "病虫害风险评估":
                prompt = f"【病虫害风险评估】基于图像分析的{count}株作物，请评估当前可能面临的病虫害风险。"
                if base_info:
                    prompt += f"\n\n农田情况：\n• " + "\n• ".join(base_info)
                prompt += "\n\n请分析：\n1. 当前种植密度和环境下最可能出现的病虫害种类\n2. 早期病虫害预警信号\n3. 预防和控制措施\n4. 是否建议使用农药，如使用应选择什么类型"
                template_prompts.append(prompt)
            
            elif template == "土壤适应性分析":
                prompt = f"【土壤适应性分析】基于图像中{count}株作物的分析，请评估当前土壤条件的适应性。"
                if base_info:
                    prompt += f"\n\n农田情况：\n• " + "\n• ".join(base_info)
                prompt += "\n\n请详细分析：\n1. 当前土壤类型是否适合种植这种作物\n2. 观察到的土壤问题或优势\n3. 土壤改良的具体建议（如需要）\n4. 适合当前土壤的肥料选择"
                template_prompts.append(prompt)
            
            elif template == "种植方式优化":
                prompt = f"【种植方式优化】根据图像中{count}株作物的分布情况，请分析当前种植方式并提供优化建议。"
                if base_info:
                    prompt += f"\n\n农田情况：\n• " + "\n• ".join(base_info)
                prompt += "\n\n请分析：\n1. 当前种植方式的优缺点\n2. 更适合这种作物的种植方式方法\n3. 如何调整行距、株距以优化生长空间\n4. 改进种植方式的具体实施步骤"
                template_prompts.append(prompt)
            
            elif template == "季节性种植建议":
                prompt = f"【季节性种植建议】基于图像中{count}株作物的分析，请提供当前季节的种植管理建议。"
                if base_info:
                    prompt += f"\n\n农田情况：\n• " + "\n• ".join(base_info)
                prompt += "\n\n请提供：\n1. 当前季节需特别注意的管理要点\n2. 应对季节性气候变化的措施\n3. 下一阶段的准备工作\n4. 优化光照、温度等环境因素的方法"
                template_prompts.append(prompt)
    
    #
    if not template_prompts:
        if language == "en":
            final_prompt = f"In the image, {count} plants were detected."
            if base_info:
                final_prompt += f"\n\nMy field conditions:\n• " + "\n• ".join(base_info)
            final_prompt += "\n\nBased on the above information, please analyze the current planting status and provide comprehensive agricultural management advice, including:\n1. Planting density evaluation\n2. Yield optimization suggestions\n3. Soil and irrigation management\n4. Prevention of potential issues"
        else:
            final_prompt = f"图像中检测到{count}株作物。"
            if base_info:
                final_prompt += f"\n\n我的农田情况：\n• " + "\n• ".join(base_info)
            final_prompt += "\n\n请根据以上信息，分析当前种植状况并提供全面的农业管理建议。包括：\n1. 种植密度评估\n2. 产量优化建议\n3. 土壤与灌溉管理\n4. 潜在问题预防"
    else:
        #
        if language == "en":
            final_prompt = f"Based on the uploaded crop image (detected approximately {count} plants), please provide professional analysis for the following aspects:\n\n" + "\n\n".join(template_prompts)
        else:
            final_prompt = f"基于上传的作物图像（检测到约{count}株），请针对以下几个方面提供专业分析：\n\n" + "\n\n".join(template_prompts)
    
    return JSONResponse(content={"prompt": final_prompt})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)