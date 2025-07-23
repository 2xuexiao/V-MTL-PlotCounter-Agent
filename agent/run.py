import uvicorn
import os
import sys

if __name__ == "__main__":
    print("Launch of the Crop Count Analysis System...")
    
    # check
    app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
    if not os.path.exists(app_dir):
        print(f"✗ Application directory does not exist: {app_dir}")
        sys.exit(1)
    
    # check
    static_dir = os.path.join(app_dir, "static")
    if not os.path.exists(static_dir):
        print(f"✓ Creating a static file directory: {static_dir}")
        os.makedirs(static_dir, exist_ok=True)
    
    for subdir in ["css", "js", "uploads", "results"]:
        subdir_path = os.path.join(static_dir, subdir)
        if not os.path.exists(subdir_path):
            print(f"✓ Creating subdirectories: {subdir_path}")
            os.makedirs(subdir_path, exist_ok=True)
    
    # try Ollama
    try:
        import httpx
        import asyncio
        
        async def check_ollama():
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get("http://127.0.0.1:11434/api/version")
                    if resp.status_code == 200:
                        data = resp.json()
                        version = data.get("version", "unknown")
                        print(f"✓ Ollama service available，version: {version}")
                        
                        # check
                        models_resp = await client.get("http://127.0.0.1:11434/api/tags")
                        if models_resp.status_code == 200:
                            models = models_resp.json().get("models", [])
                            model_names = [m.get("name") for m in models]
                            print(f"✓ models: {', '.join(model_names)}")
                            
                            if "gemma" not in str(model_names).lower():
                                print("⚠ warning: No gemma model detected. You may need to run: ollama pull gemma")
                    else:
                        print(f"✗ Ollama API return error: {resp.status_code}")
            except Exception as e:
                print(f"✗ Ollama service detection fail: {str(e)}")
                print("⚠ Warning: If you want to use the Ollama model functionality, please ensure that the Ollama service is running.")
                print("⚠ Start the service with the command 'ollama serve', then download the model using 'ollama pull gemma'")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(check_ollama())
        loop.close()
    except Exception as e:
        print(f"✗ Ollama service Detection Error: {str(e)}")
    
    # Start the application
    print("\n✓ Start the application")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8010, reload=True)