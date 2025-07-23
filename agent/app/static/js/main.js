// DOM Element Selection
const uploadSection = document.getElementById('upload-section');
const resultsSection = document.getElementById('results-section');
const uploadContainer = document.querySelector('.upload-container');
const fileInput = document.getElementById('file-input');
const resultImage = document.getElementById('result-image');
const countNumber = document.getElementById('total-count');
const chatMessages = document.querySelector('.chat-messages');
const chatInput = document.getElementById('chat-input');
const sendButton = document.querySelector('.send-button');
const modalOverlay = document.getElementById('modal-overlay');
const modalOptionsContainer = document.getElementById('modal-options-container');
const generatePromptButton = document.getElementById('generate-prompt');
const cancelOptionsButton = document.getElementById('cancel-options');
const generateFromOptionsButton = document.getElementById('generate-from-options');
const templateOptionsContainer = document.getElementById('modal-template-options');
// Add image preview elements
const imageViewerOverlay = document.getElementById('image-viewer-overlay');
const imageViewerImg = document.getElementById('image-viewer-img');
const imageViewerClose = document.getElementById('image-viewer-close');
const zoomInBtn = document.getElementById('zoom-in');
const zoomOutBtn = document.getElementById('zoom-out');
const resetZoomBtn = document.getElementById('reset-zoom');
const previewableImages = document.querySelectorAll('.preview-enabled');
const viewFullImageBtn = document.querySelector('.view-full-image-btn');

// Navigation bar and help elements
const helpButton = document.getElementById('help-button');
const helpOverlay = document.getElementById('help-overlay');
const helpClose = document.getElementById('help-close');
const helpGotItButton = document.getElementById('help-got-it');
const languageOptions = document.querySelectorAll('.language-option');

// Elements related to options
const cropTypeSelect = document.getElementById('modal-crop-type');
const soilTypeSelect = document.getElementById('modal-soil-type');
const irrigationSelect = document.getElementById('modal-irrigation');
const growthStageSelect = document.getElementById('modal-growth-stage');
const seasonSelect = document.getElementById('modal-season');
const climateSelect = document.getElementById('modal-climate');
const plantingMethodSelect = document.getElementById('modal-planting-method');
const fieldSizeSelect = document.getElementById('modal-field-size');

// Create a typing effect indicator element
const typingIndicator = document.createElement('div');
typingIndicator.className = 'typing-indicator';
typingIndicator.innerHTML = '<span></span><span></span><span></span>';
typingIndicator.style.display = 'none';

// Save option data
let optionsData = null;
let selectedTemplates = [];

// Current language
let currentLanguage = 'en';

// Multilingual text
const translations = {
    zh: {
        navTitle: '作物计数智能体',
        helpText: '帮助',
        languageText: '语言',
        helpTitle: '系统使用帮助',
        helpSection1: '系统概述',
        helpOverview: '作物计数智能体是一个帮助农业人员对上传的农田图像进行分析的工具。本系统可以识别图像中的作物，计算作物数量，并提供详细的生长分析和建议。',
        helpSection2: '使用流程',
        helpStep1: '在左侧上传区域点击或拖放图片/视频文件（支持JPG、PNG、MP4格式）',
        helpStep2: '系统自动处理图像并在左侧显示计数密度图和作物数量',
        helpStep3: '点击密度图可以查看大图并进行放大缩小操作',
        helpStep4: '点击"生成分析提示词"按钮，选择作物信息和分析类型',
        helpStep5: '在弹出的选项中选择相关农业参数，如作物类型、土壤类型等',
        helpStep6: '系统生成针对您上传图像的智能分析，并在右侧聊天框显示',
        helpStep7: '您可以继续在聊天框提问，获取更详细的分析和建议',
        helpSection3: '注意事项',
        helpNote1: '为了获得最佳分析结果，请上传清晰的农田图像，尽量避免过暗、过亮或模糊的图像',
        helpNote2: '系统分析结果仅供参考，最终农业决策应结合实际情况',
        helpNote3: '如有任何问题或建议，请联系系统管理员',
        helpGotIt: '我知道了',
        uploadPrompt: '点击或拖拽上传图片/视频',
        supportedFormats: '支持格式: JPG, PNG, MP4',
        sectionTitle: '计数密度图',
        countLabel: '计数结果',
        uploadNew: '上传新图片',
        generatePrompt: '生成分析提示词',
        chatHeader: 'AI 分析助手',
        chatPlaceholder: '输入您的问题，或点击"生成分析提示词"按钮自动生成分析请求...',
        systemWelcome: '您好！我是AI分析助手。请上传图片，我会协助您分析作物生长情况。',
        imageProcessed: '图片分析完成！您可以询问关于分析结果的问题。',
        processingError: '图片处理完成，但显示结果时出现问题。您仍然可以询问有关图像的问题。',
        optionsTitle: '选择作物信息和分析类型',
        selectAnalysisType: '选择分析类型 (可多选):',
        cropType: '作物类型:',
        soilType: '土壤类型:',
        irrigation: '灌溉方式:',
        growthStage: '生长阶段:',
        season: '季节:',
        climate: '气候条件:',
        plantingMethod: '种植方式:',
        fieldSize: '田地大小:',
        cancel: '取消',
        generate: '生成提示词',
        selectAtLeastOne: '请至少选择一个分析类型',
        uploadFailed: '上传失败，请重试',
        generateFailed: '生成提示词失败，请重试',
        loadOptionsFailed: '加载选项失败，请重试',
        //
        analysisReport: '【图像分析报告】',
        analysisComplete: '我已完成对上传图像的分析，主要数据如下：',
        detectedCrops: '检测到的作物:',
        cropType: '作物类型',
        densityMetric: '密度指标:',
        perTenThousand: '每万像素约',
        imageSize: '图像尺寸:',
        unknown: '未知',
        plants: '株',
        askQuestions: '您可以询问我以下问题:',
        densitySuitability: '这个密度是否适合某种作物生长？',
        optimizePlanting: '如何优化种植方式提高产量？',
        irrigationSuggestion: '针对当前密度的灌溉或土壤改良建议',
        pestRisk: '请提供该地块的播种详情，我将分析出苗率',
        clickGenerate: '或者点击',
        detailedAnalysis: '按钮获取更详细的分析。',
        //
        analysisTypes: {
            '种植密度分析': '种植密度分析',
            '产量优化建议': '产量优化建议',
            '灌溉方案建议': '灌溉方案建议',
            '病虫害风险评估': '病虫害风险评估',
            '土壤适应性分析': '土壤适应性分析',
            '种植方式优化': '种植方式优化',
            '季节性种植建议': '季节性种植建议'
        },
        pleaseSelect: '请选择...',
        viewFullImage: '点击查看大图',
        closePreview: '关闭',
    },
    en: {
        navTitle: 'V-MTL-PlotCounter Agent',
        helpText: 'Help',
        languageText: 'Language',
        helpTitle: 'System Usage Guide',
        helpSection1: 'System Overview',
        helpOverview: 'The PlotCounter Agent is a tool that helps agricultural personnel analyze uploaded field images. The system can identify crops in images, count them, and provide detailed growth analysis and recommendations.',
        helpSection2: 'Usage Process',
        helpStep1: 'Click or drag and drop image/video files in the upload area on the left (supported formats: JPG, PNG, MP4)',
        helpStep2: 'The system automatically processes the image and displays the count map and crop count on the left',
        helpStep3: 'Click on the density map to view the full-size image and zoom in/out',
        helpStep4: 'Click the "Generate Analysis Prompt" button and select crop information and analysis types',
        helpStep5: 'Select relevant agricultural parameters in the pop-up options, such as crop type, soil type, etc.',
        helpStep6: 'The system generates intelligent analysis for your uploaded image and displays it in the chat box on the right',
        helpStep7: 'You can continue to ask questions in the chat box for more detailed analysis and recommendations',
        helpSection3: 'Notes',
        helpNote1: 'For the best analysis results, please upload clear field images, avoiding overly dark, bright, or blurry images',
        helpNote2: 'System analysis results are for reference only, final agricultural decisions should be made based on actual conditions',
        helpNote3: 'If you have any questions or suggestions, please contact the system administrator',
        helpGotIt: 'Got it',
        uploadPrompt: 'Click or drag to upload image/video',
        supportedFormats: 'Supported formats: JPG, PNG, MP4',
        sectionTitle: 'Count Map',
        countLabel: 'Count Result',
        uploadNew: 'Upload New Image',
        generatePrompt: 'Generate Analysis Prompt',
        chatHeader: 'AI Analysis Assistant',
        chatPlaceholder: 'Enter your question, or click the "Generate Analysis Prompt" button to automatically generate an analysis request...',
        systemWelcome: 'Hello! I am the AI Analysis Assistant. Please upload an image, and I will help you analyze the crop growth situation.',
        imageProcessed: 'Image analysis complete! You can ask questions about the analysis results.',
        processingError: 'Image processing complete, but there was a problem displaying the results. You can still ask questions about the image.',
        optionsTitle: 'Select Crop Information and Analysis Type',
        selectAnalysisType: 'Select Analysis Type (Multiple):', 
        cropType: 'Crop Type:',
        soilType: 'Soil Type:',
        irrigation: 'Irrigation:',
        growthStage: 'Growth Stage:',
        season: 'Season:',
        climate: 'Climate Conditions:',
        plantingMethod: 'Planting Method:',
        fieldSize: 'Field Size:',
        cancel: 'Cancel',
        generate: 'Generate Prompt',
        selectAtLeastOne: 'Please select at least one analysis type',
        uploadFailed: 'Upload failed, please try again',
        generateFailed: 'Failed to generate prompt, please try again',
        loadOptionsFailed: 'Failed to load options, please try again',
        // Analysis report related
        analysisReport: '[Image Analysis Report]',
        analysisComplete: 'I have completed the analysis of the uploaded image. Here are the main findings:',
        detectedCrops: 'Predicted seeding count:',
        cropType: 'Variety',
        densityMetric: 'Density metric:',
        perTenThousand: 'approx. per 10,000 pixels',
        imageSize: 'Image size:',
        unknown: 'unknown',
        plants: 'plants',
        askQuestions: 'You can ask me the following questions:',
        densitySuitability: 'Is this density suitable for a specific crop growth?',
        optimizePlanting: 'How to optimize planting methods to increase yield?',
        irrigationSuggestion: 'Irrigation or soil improvement suggestions for the current density',
        pestRisk: 'Please provide sowing details for the plot, and I\'ll analyze the emergence rate.',
        clickGenerate: 'Or click the',
        detailedAnalysis: 'button for more detailed analysis.',
        // Option related translations
        analysisTypes: {
            '种植密度分析': 'Planting Density Analysis',
            '产量优化建议': 'Yield Optimization Suggestions',
            '灌溉方案建议': 'Irrigation Plan Recommendations',
            '病虫害风险评估': 'Pest & Disease Risk Assessment',
            '土壤适应性分析': 'Soil Adaptability Analysis',
            '种植方式优化': 'Planting Method Optimization',
            '季节性种植建议': 'Seasonal Planting Recommendations'
        },
        pleaseSelect: 'Please select...',
        viewFullImage: 'Click to view full image',
        closePreview: 'Close',
    }
};

// Initialize page state
document.addEventListener('DOMContentLoaded', () => {
    resultsSection.style.display = 'none';
    
    // Bind image preview function
    initializeImageViewer();

    if (generatePromptButton) {
        generatePromptButton.addEventListener('click', showOptionsPanel);
    }

    if (cancelOptionsButton) {
        cancelOptionsButton.addEventListener('click', hideOptionsPanel);
    }

    if (generateFromOptionsButton) {
        generateFromOptionsButton.addEventListener('click', generatePromptFromOptions);
    }

    modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) {
            hideOptionsPanel();
        }
    });

    if (helpButton) {
        helpButton.addEventListener('click', showHelpPanel);
    }

    if (helpClose) {
        helpClose.addEventListener('click', hideHelpPanel);
    }

    if (helpGotItButton) {
        helpGotItButton.addEventListener('click', hideHelpPanel);
    }

    languageOptions.forEach(option => {
        option.addEventListener('click', () => {
            languageOptions.forEach(op => op.classList.remove('active'));
            option.classList.add('active');
            changeLanguage(option.getAttribute('data-lang'));
        });
    });

    languageOptions.forEach(option => {
        if (option.getAttribute('data-lang') === currentLanguage) {
            option.classList.add('active');
        } else {
            option.classList.remove('active');
        }
    });

    applyTranslations();

    const hasVisited = localStorage.getItem('hasVisitedBefore');
    if (!hasVisited) {
        setTimeout(showHelpPanel, 500);
        localStorage.setItem('hasVisitedBefore', 'true');
    }
});

function showHelpPanel() {
    helpOverlay.classList.add('active');
}

function hideHelpPanel() {
    helpOverlay.classList.remove('active');
}

function changeLanguage(lang) {
    currentLanguage = lang;
    applyTranslations();
}

function applyTranslations() {
    const t = translations[currentLanguage];

    document.getElementById('nav-title').textContent = t.navTitle;
    document.getElementById('help-text').textContent = t.helpText;
    document.getElementById('language-text').textContent = t.languageText;

    document.getElementById('help-title').textContent = t.helpTitle;
    document.getElementById('help-section-1').textContent = t.helpSection1;
    document.getElementById('help-overview').textContent = t.helpOverview;
    document.getElementById('help-section-2').textContent = t.helpSection2;
    document.getElementById('help-step-1').textContent = t.helpStep1;
    document.getElementById('help-step-2').textContent = t.helpStep2;
    document.getElementById('help-step-3').textContent = t.helpStep3;
    document.getElementById('help-step-4').textContent = t.helpStep4;
    document.getElementById('help-step-5').textContent = t.helpStep5;
    document.getElementById('help-step-6').textContent = t.helpStep6;
    document.getElementById('help-step-7').textContent = t.helpStep7;
    document.getElementById('help-section-3').textContent = t.helpSection3;
    document.getElementById('help-note-1').textContent = t.helpNote1;
    document.getElementById('help-note-2').textContent = t.helpNote2;
    document.getElementById('help-note-3').textContent = t.helpNote3;
    document.getElementById('help-got-it').textContent = t.helpGotIt;

    if (document.querySelector('.upload-prompt p')) {
        document.querySelector('.upload-prompt p').textContent = t.uploadPrompt;
    }
    if (document.querySelector('.supported-formats')) {
        document.querySelector('.supported-formats').textContent = t.supportedFormats;
    }

    if (document.querySelector('.section-title')) {
        document.querySelector('.section-title').textContent = t.sectionTitle;
    }
    if (document.querySelector('.count-label')) {
        document.querySelector('.count-label').textContent = t.countLabel;
    }

    if (document.getElementById('upload-new')) {
        const uploadNewButton = document.getElementById('upload-new');
        uploadNewButton.innerHTML = `<i class="fas fa-upload"></i> ${t.uploadNew}`;
    }
    if (generatePromptButton) {
        generatePromptButton.innerHTML = `<i class="fas fa-magic"></i> ${t.generatePrompt}`;
    }

    if (document.querySelector('.chat-header h3')) {
        document.querySelector('.chat-header h3').innerHTML = `<i class="fas fa-robot"></i> ${t.chatHeader}`;
    }
    if (chatInput) {
        chatInput.placeholder = t.chatPlaceholder;
    }

    const firstMessage = document.querySelector('.system-message .message-content');
    if (firstMessage && firstMessage.textContent.trim() === translations.zh.systemWelcome) {
        firstMessage.textContent = t.systemWelcome;
    }

    if (document.querySelector('.options-title')) {
        document.querySelector('.options-title').textContent = t.optionsTitle;
    }
    if (document.querySelector('.template-selection label')) {
        document.querySelector('.template-selection label').textContent = t.selectAnalysisType;
    }

    const optionLabels = document.querySelectorAll('.option-label');
    optionLabels.forEach(label => {
        if (label.textContent === '作物类型:') label.textContent = t.cropType;
        if (label.textContent === '土壤类型:') label.textContent = t.soilType;
        if (label.textContent === '灌溉方式:') label.textContent = t.irrigation;
        if (label.textContent === '生长阶段:') label.textContent = t.growthStage;
        if (label.textContent === '季节:') label.textContent = t.season;
        if (label.textContent === '气候条件:') label.textContent = t.climate;
        if (label.textContent === '种植方式:') label.textContent = t.plantingMethod;
        if (label.textContent === '田地大小:') label.textContent = t.fieldSize;
    });

    if (document.getElementById('cancel-options')) {
        document.getElementById('cancel-options').textContent = t.cancel;
    }
    if (document.getElementById('generate-from-options')) {
        document.getElementById('generate-from-options').textContent = t.generate;
    }

    if (viewFullImageBtn) {
        viewFullImageBtn.textContent = t.viewFullImage;
    }
}

uploadContainer.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => processFile(e.target.files[0]));

uploadContainer.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadContainer.style.borderColor = '#2E7D32';
    uploadContainer.style.backgroundColor = 'rgba(76, 175, 80, 0.1)';
});

uploadContainer.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadContainer.style.borderColor = '';
    uploadContainer.style.backgroundColor = '';
});

uploadContainer.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadContainer.style.borderColor = '';
    uploadContainer.style.backgroundColor = '';
    const file = e.dataTransfer.files[0];
    processFile(file);
});

function processFile(file) {
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    uploadFile(formData);
}

async function uploadFile(formData) {
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        const t = translations[currentLanguage];
        alert(t.uploadFailed);
    }
}

// Image preview function
let currentZoom = 1;
const zoomStep = 0.2;

function openImageViewer(imageSrc) {
    if (!imageSrc) return;
    
    imageViewerImg.src = imageSrc;
    imageViewerOverlay.style.display = 'flex';
    document.body.style.overflow = 'hidden';
    currentZoom = 1;
    resetZoom();

    document.addEventListener('keydown', handleKeydown);
}

function closeImageViewer() {
    imageViewerOverlay.style.display = 'none';
    document.body.style.overflow = '';

    document.removeEventListener('keydown', handleKeydown);
}

function handleKeydown(e) {
    if (e.key === 'Escape') {
        closeImageViewer();
    } else if (e.key === '+' || e.key === '=') {
        zoomIn();
    } else if (e.key === '-' || e.key === '_') {
        zoomOut();
    } else if (e.key === '0') {
        resetZoom();
    }
}

function zoomIn() {
    currentZoom += zoomStep;
    applyZoom();
}

function zoomOut() {
    currentZoom = Math.max(0.2, currentZoom - zoomStep);
    applyZoom();
}

function resetZoom() {
    currentZoom = 1;
    applyZoom();
}

function applyZoom() {
    imageViewerImg.style.transform = `scale(${currentZoom})`;
}

// Initialize the preview function
function initializeImageViewer() {
    if (imageViewerClose) {
        imageViewerClose.addEventListener('click', closeImageViewer);
    }

    if (zoomInBtn) {
        zoomInBtn.addEventListener('click', zoomIn);
    }
    
    if (zoomOutBtn) {
        zoomOutBtn.addEventListener('click', zoomOut);
    }
    
    if (resetZoomBtn) {
        resetZoomBtn.addEventListener('click', resetZoom);
    }

    previewableImages.forEach(img => {
        img.addEventListener('click', () => {
            openImageViewer(img.src);
        });
    });

    if (viewFullImageBtn) {
        viewFullImageBtn.addEventListener('click', () => {
            if (resultImage && resultImage.src) {
                openImageViewer(resultImage.src);
            }
        });
    }

    imageViewerOverlay.addEventListener('click', (e) => {
        if (e.target === imageViewerOverlay) {
            closeImageViewer();
        }
    });
}

// Display result function
function displayResults(data) {
    try {
        console.log("Received data:", data);

        uploadSection.style.display = 'none';
        resultsSection.style.display = 'flex';

        if (data.result_image) {
            resultImage.src = data.result_image;
            console.log("Set the image path:", data.result_image);
        } else {
            console.error("The 'result_image' field was not found.");
        }
        
        if (data.count !== undefined && countNumber) {
            countNumber.textContent = data.count;
            console.log("Set the count:", data.count);
        } else {
            console.error("The 'count' field was not found or the 'countNumber' element does not exist.");
        }

        const t = translations[currentLanguage];
        addMessage(t.imageProcessed, 'system');

        if (viewFullImageBtn) {
            viewFullImageBtn.textContent = translations[currentLanguage].viewFullImage;
        }

        if (data.count !== undefined) {
            try {
                const analysisCardHTML = createAnalysisCard(data);
                addMessage(analysisCardHTML, 'ai');

                const historyMessage = currentLanguage === 'zh' ? 
                    `图像分析完成，检测到${data.count}株作物。` : 
                    `Image analysis complete, detected ${data.count} plants.`;
                
                messageHistory.push({
                    role: 'assistant',
                    content: historyMessage
                });
            } catch (cardError) {
                console.error('创建分析卡片时出错:', cardError);
                const fallbackMessage = currentLanguage === 'zh' ? 
                    `图像分析完成，检测到${data.count}株作物。您可以询问有关分析结果的问题。` : 
                    `Image analysis complete, detected ${data.count} plants. You can ask questions about the analysis results.`;
                
                addMessage(fallbackMessage, 'ai');
            }
        }

        if (resultImage) {
            if (data.result_image) {
                resultImage.src = data.result_image + '?t=' + new Date().getTime();
            }
            else if (data.density_map_path) {
                resultImage.src = data.density_map_path + '?t=' + new Date().getTime();
            }

            resultImage.onload = function() {
                resultImage.classList.add('preview-enabled');
                initializeImageViewer();
                console.log("The image has finished loading, and the preview function has been enabled.");
            };

            resultImage.onerror = function() {
                console.error("Image loading failed:", resultImage.src);
                resultImage.src = '/static/img/image-error.png';
            };
        }
    } catch (error) {
        console.error('An error occurred while displaying the result:', error);
        uploadSection.style.display = 'none';
        resultsSection.style.display = 'flex';
        const t = translations[currentLanguage];
        addMessage(t.processingError, 'system');
    }
}

// Create an analysis result card
function createAnalysisCard(data) {
    const density = data.density_per_unit ? data.density_per_unit.toFixed(2) : "未知";
    const t = translations[currentLanguage];
    let imageDimensions = data.image_dimensions || t.unknown;
    if (imageDimensions && imageDimensions.includes('x')) {
        imageDimensions = imageDimensions.replace('x', ' x ');
    }

    let cropType = t.unknown;
    if (currentLanguage === 'zh') {
        cropType = data.crop_type || t.unknown;
    } else {
        cropType = data.crop_type_en || t.unknown;
    }
    
    return `
    <div class="info-card highlight-card">
        <div class="info-card-title">${t.analysisReport}</div>
        <div class="info-card-content">
            <p>${t.analysisComplete}</p>
            <ul>
                <li><strong>${t.detectedCrops}</strong> ${data.count} ${t.plants}</li>
                <li><strong>${t.cropType || '作物类型'}:</strong> ${cropType}</li>
                <li><strong>${t.imageSize}</strong> ${imageDimensions}</li>
            </ul>
            <p>${t.askQuestions}</p>
            <ul>
                <li>${t.densitySuitability}</li>
                <li>${t.optimizePlanting}</li>
                <li>${t.irrigationSuggestion}</li>
                <li>${t.pestRisk}</li>
            </ul>
            <p>${t.clickGenerate} <strong>${t.generatePrompt}</strong> ${t.detailedAnalysis}</p>
        </div>
    </div>`;
}

function resetUpload() {
    fileInput.value = '';
    resultsSection.style.display = 'none';
    uploadSection.style.display = 'flex';
}

let messageHistory = [];

chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendButton.addEventListener('click', sendMessage);

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    addUserMessage(message);
    chatInput.value = '';

    const typingMsgDiv = document.createElement('div');
    typingMsgDiv.className = 'message ai-message typing';
    typingMsgDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    chatMessages.appendChild(typingMsgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                message, 
                history: messageHistory,
                language: currentLanguage
            })
        });

        if (typingMsgDiv.parentNode) {
            chatMessages.removeChild(typingMsgDiv);
        }

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        addAIMessage(data.response);

        messageHistory.push({
            role: 'user',
            content: message
        });
        
        messageHistory.push({
            role: 'assistant',
            content: data.response
        });
    } catch (error) {
        console.error('Error:', error);
        if (typingMsgDiv.parentNode) {
            chatMessages.removeChild(typingMsgDiv);
        }

        const errorMsg = currentLanguage === 'zh' ? 
            "很抱歉，处理您的请求时出现了错误。请重试或检查网络连接。" : 
            "Sorry, an error occurred while processing your request. Please try again or check your network connection.";
        
        addMessage(errorMsg, 'system');
    }
}

// Add chat message to interface
function addMessage(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const formattedContent = formatMessageContent(content);
    contentDiv.innerHTML = formattedContent;

    const timeSpan = document.createElement('span');
    timeSpan.className = 'message-time';
    const now = new Date();
    timeSpan.textContent = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;

    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timeSpan);

    chatMessages.appendChild(messageDiv);

    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Format function
function formatText(text) {
    if (!text) return '';

    let formattedText = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    formattedText = formattedText.replace(/\*(.*?)\*/g, '<em>$1</em>');

    formattedText = formattedText.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    
    return formattedText;
}

function formatMessageContent(content) {
    if (!content) return '';

    content = content.replace(/```([\s\S]*?)```/g, function(match, code) {
        return `<div class="code-block"><code>${code.trim()}</code></div>`;
    });

    let lines = content.split('\n');
    let result = [];

    let inOrderedList = false;
    let inUnorderedList = false;
    let currentListType = null;
    let listIndentLevel = 0;
    let nestedListStacks = [];

    for (let i = 0; i < lines.length; i++) {
        let line = lines[i];

        if (line.match(/^(#{1,6})\s+(.+)$/)) {
            if (inOrderedList || inUnorderedList) {
                result.push(closeCurrentList(currentListType));
                inOrderedList = false;
                inUnorderedList = false;
                currentListType = null;
                nestedListStacks = [];
            }
            
            let headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
            let level = headingMatch[1].length;
            let text = formatText(headingMatch[2]);
            result.push(`<h${level} class="content-heading">${text}</h${level}>`);
            continue;
        }

        if (line.match(/^---+$/)) {
            if (inOrderedList || inUnorderedList) {
                result.push(closeCurrentList(currentListType));
                inOrderedList = false;
                inUnorderedList = false;
                currentListType = null;
                nestedListStacks = [];
            }
            result.push('<hr>');
            continue;
        }

        if (line.match(/^>(info|suggestion|highlight):\s*(.+)$/i)) {
            if (inOrderedList || inUnorderedList) {
                result.push(closeCurrentList(currentListType));
                inOrderedList = false;
                inUnorderedList = false;
                currentListType = null;
                nestedListStacks = [];
            }
            
            let cardMatch = line.match(/^>(info|suggestion|highlight):\s*(.+)$/i);
            let cardType = cardMatch[1].toLowerCase();
            let cardTitle = formatText(cardMatch[2]);
            
            let nextLine = i + 1 < lines.length ? lines[i + 1] : '';
            let cardContent = '';

            while (i + 1 < lines.length && lines[i + 1].startsWith('>') && !lines[i + 1].match(/^>(info|suggestion|highlight):/i)) {
                cardContent += formatText(lines[i + 1].substring(1).trim()) + ' '; // 格式化卡片内容
                i++;
            }
            
            result.push(`<div class="info-card ${cardType}-card">
                <div class="info-card-title">${cardTitle}</div>
                <div class="info-card-content">${cardContent.trim()}</div>
            </div>`);
            continue;
        }

        if (line.startsWith('>')) {
            if (inOrderedList || inUnorderedList) {
                result.push(closeCurrentList(currentListType));
                inOrderedList = false;
                inUnorderedList = false;
                currentListType = null;
                nestedListStacks = [];
            }
            
            let blockquoteContent = formatText(line.substring(1).trim()); // 格式化引用内容
            let j = i + 1;

            while (j < lines.length && lines[j].startsWith('>')) {
                blockquoteContent += ' ' + formatText(lines[j].substring(1).trim());
                j++;
            }
            
            i = j - 1;
            result.push(`<blockquote>${blockquoteContent}</blockquote>`);
            continue;
        }

        let orderedListMatch = line.match(/^(\s*)(\d+)\.\s+(.+)$/);
        if (orderedListMatch) {
            let indentation = orderedListMatch[1].length;
            let itemContent = formatText(orderedListMatch[3]); // 格式化列表项内容

            let thisIndentLevel = Math.floor(indentation / 2);

            if (!inOrderedList && !inUnorderedList) {
                inOrderedList = true;
                currentListType = 'ol';
                listIndentLevel = thisIndentLevel;
                result.push(`<ol class="content-list">`);
                nestedListStacks = [{ type: 'ol', level: thisIndentLevel }];
            }
            else if (thisIndentLevel > listIndentLevel) {
                result.push(`<li><ol class="nested-list">`);
                nestedListStacks.push({ type: 'ol', level: thisIndentLevel });
                listIndentLevel = thisIndentLevel;
            }
            else if (thisIndentLevel < listIndentLevel) {
                while (nestedListStacks.length > 0 && nestedListStacks[nestedListStacks.length - 1].level > thisIndentLevel) {
                    let poppedList = nestedListStacks.pop();
                    result.push(`</ol></li>`);
                }
                listIndentLevel = thisIndentLevel;
            }

            result.push(`<li>${itemContent}`);

            let j = i + 1;
            let appendContent = '';
            
            while (j < lines.length) {
                let nextLine = lines[j];
                let nextOrderedMatch = nextLine.match(/^(\s*)(\d+)\.\s+(.+)$/);
                let nextUnorderedMatch = nextLine.match(/^(\s*)[*\-+]\s+(.+)$/);

                if (!nextOrderedMatch && !nextUnorderedMatch && nextLine.trim() !== '') {
                    appendContent += ' ' + formatText(nextLine.trim());
                    j++;
                } else {
                    break;
                }
            }
            
            if (appendContent) {
                result[result.length - 1] += appendContent;
                i = j - 1;
            }
            
            continue;
        }

        let unorderedListMatch = line.match(/^(\s*)[*\-+]\s+(.+)$/);
        if (unorderedListMatch) {
            let indentation = unorderedListMatch[1].length;
            let itemContent = formatText(unorderedListMatch[2]); // 格式化列表项内容

            let thisIndentLevel = Math.floor(indentation / 2);

            if (!inOrderedList && !inUnorderedList) {
                inUnorderedList = true;
                currentListType = 'ul';
                listIndentLevel = thisIndentLevel;
                result.push(`<ul class="content-list">`);
                nestedListStacks = [{ type: 'ul', level: thisIndentLevel }];
            }
            else if (thisIndentLevel > listIndentLevel) {
                result.push(`<li><ul class="nested-list">`);
                nestedListStacks.push({ type: 'ul', level: thisIndentLevel });
                listIndentLevel = thisIndentLevel;
            }
            else if (thisIndentLevel < listIndentLevel) {
                while (nestedListStacks.length > 0 && nestedListStacks[nestedListStacks.length - 1].level > thisIndentLevel) {
                    let poppedList = nestedListStacks.pop();
                    result.push(`</ul></li>`);
                }
                listIndentLevel = thisIndentLevel;
            }

            result.push(`<li>${itemContent}`);

            let j = i + 1;
            let appendContent = '';
            
            while (j < lines.length) {
                let nextLine = lines[j];
                let nextOrderedMatch = nextLine.match(/^(\s*)(\d+)\.\s+(.+)$/);
                let nextUnorderedMatch = nextLine.match(/^(\s*)[*\-+]\s+(.+)$/);

                if (!nextOrderedMatch && !nextUnorderedMatch && nextLine.trim() !== '') {
                    appendContent += ' ' + formatText(nextLine.trim());
                    j++;
                } else {
                    break;
                }
            }
            
            if (appendContent) {
                result[result.length - 1] += appendContent;
                i = j - 1;
            }
            
            continue;
        }

        if (line.trim() === '') {
            if (i + 1 < lines.length) {
                let nextLine = lines[i + 1];
                let isNextLineList = nextLine.match(/^(\s*)(\d+)\.\s+(.+)$/) || nextLine.match(/^(\s*)[*\-+]\s+(.+)$/);
                
                if (!isNextLineList && (inOrderedList || inUnorderedList)) {
                    result.push(closeCurrentList(currentListType));
                    inOrderedList = false;
                    inUnorderedList = false;
                    currentListType = null;
                    nestedListStacks = [];
                }
            }
            result.push('<p></p>');
            continue;
        }

        if (inOrderedList || inUnorderedList) {
            result.push(`</li>`);
        } else {
            line = formatText(line);
            result.push(`<p>${line}</p>`);
        }
    }

    if (inOrderedList || inUnorderedList) {
        result.push(closeCurrentList(currentListType));
    }
    
    // Helper function: Close the current list and all its nested lists
    function closeCurrentList(listType) {
        if (!listType) return '';
        
        let closingTags = '';
        for (let i = nestedListStacks.length - 1; i >= 0; i--) {
            if (i === nestedListStacks.length - 1) {
                closingTags += `</li></${nestedListStacks[i].type}>`;
            } else {
                closingTags += `</li></${nestedListStacks[i].type}>`;
            }
        }
        return closingTags;
    }
    
    return result.join('');
}

// Add user message
function addUserMessage(message) {
    addMessage(message, 'user');
    messageHistory.push({
        role: 'user',
        content: message
    });
}

// Add AI message
function addAIMessage(message) {
    addMessage(message, 'ai');
    messageHistory.push({
        role: 'assistant',
        content: message
    });
}

// Generate prompt
async function generatePrompt() {
    try {
        const response = await fetch('/generate_prompt', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        chatInput.value = data.prompt;
        chatInput.focus();
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to generate prompt, please try again.');
    }
}

// Show options panel
async function showOptionsPanel() {
    try {
        const response = await fetch('/generate_prompt', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                language: currentLanguage
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        optionsData = await response.json();

        populateSelectOptions(cropTypeSelect, optionsData.options.crop_types);
        populateSelectOptions(soilTypeSelect, optionsData.options.soil_types);
        populateSelectOptions(irrigationSelect, optionsData.options.irrigation_methods);
        populateSelectOptions(growthStageSelect, optionsData.options.growth_stages);
        populateSelectOptions(seasonSelect, optionsData.options.seasons);
        populateSelectOptions(climateSelect, optionsData.options.climate_conditions);
        populateSelectOptions(plantingMethodSelect, optionsData.options.planting_methods);
        populateSelectOptions(fieldSizeSelect, optionsData.options.field_sizes);

        createTemplateOptions(optionsData.templates);

        modalOverlay.style.display = 'flex';

        selectedTemplates = [];
    } catch (error) {
        console.error('Error:', error);
        const t = translations[currentLanguage];
        alert(t.loadOptionsFailed);
    }
}

// Hide options panel
function hideOptionsPanel() {
    modalOverlay.style.display = 'none';
}

// Populate select options
function populateSelectOptions(selectElement, options) {
    if (!selectElement) return;
    
    selectElement.innerHTML = '';

    const t = translations[currentLanguage];
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = t.pleaseSelect;
    selectElement.appendChild(defaultOption);

    if (Array.isArray(options)) {
        options.forEach(option => {
            const optionElement = document.createElement('option');
            optionElement.value = option;
            optionElement.textContent = option;
            selectElement.appendChild(optionElement);
        });
    }
}

// Create template options
function createTemplateOptions(templates) {
    if (!templateOptionsContainer) return;
    
    templateOptionsContainer.innerHTML = '';
    
    if (Array.isArray(templates)) {
        templates.forEach(template => {
            const templateButton = document.createElement('div');
            templateButton.className = 'template-option';

            const t = translations[currentLanguage];
            const displayText = currentLanguage === 'en' && t.analysisTypes[template] ? 
                                t.analysisTypes[template] : template;
            
            templateButton.textContent = displayText;
            templateButton.dataset.originalValue = template;
            
            templateButton.addEventListener('click', () => {
                if (templateButton.classList.contains('selected')) {
                    templateButton.classList.remove('selected');
                    selectedTemplates = selectedTemplates.filter(t => t !== template);
                } else {
                    templateButton.classList.add('selected');
                    selectedTemplates.push(template);
                }
            });
            
            templateOptionsContainer.appendChild(templateButton);
        });
    }
}

// Generate prompt from options
async function generatePromptFromOptions() {
    if (selectedTemplates.length === 0) {
        const t = translations[currentLanguage];
        alert(t.selectAtLeastOne);
        return;
    }

    const userOptions = {
        crop_type: cropTypeSelect ? cropTypeSelect.value : '',
        soil_type: soilTypeSelect ? soilTypeSelect.value : '',
        irrigation: irrigationSelect ? irrigationSelect.value : '',
        growth_stage: growthStageSelect ? growthStageSelect.value : '',
        season: seasonSelect ? seasonSelect.value : '',
        climate: climateSelect ? climateSelect.value : '',
        planting_method: plantingMethodSelect ? plantingMethodSelect.value : '',
        field_size: fieldSizeSelect ? fieldSizeSelect.value : '',
        language: currentLanguage
    };
    
    try {
        const response = await fetch('/generate_prompt_from_options', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                templates: selectedTemplates,
                options: userOptions,
                language: currentLanguage
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        chatInput.value = data.prompt;
        chatInput.focus();

        hideOptionsPanel();
    } catch (error) {
        console.error('Error:', error);
        const t = translations[currentLanguage];
        alert(t.generateFailed);
    }
}