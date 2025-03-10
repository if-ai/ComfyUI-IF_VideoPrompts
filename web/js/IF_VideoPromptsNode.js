import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
    name: "IF_VideoPrompts",
    async setup() {
        console.log("Video Prompt Generator extension setup");
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "VideoPromptNode") {
            // Enhance the node's appearance and functionality
            
            // Override the onDrawBackground method to add processing information
            const origDrawBackground = nodeType.prototype.onDrawBackground;
            nodeType.prototype.onDrawBackground = function(ctx) {
                if (origDrawBackground) {
                    origDrawBackground.apply(this, arguments);
                }
                
                if (this.widgets && this.widgets.length > 0) {
                    // Display frame count if images are connected
                    const images = this.getInputData(0);
                    if (images && images.shape) {
                        let frameCount = images.shape[0];
                        ctx.fillStyle = "#AAA";
                        ctx.font = "12px Arial";
                        ctx.fillText(`Sequence: ${frameCount} frame(s)`, 10, 35);
                        
                        // Display a visual indication of frame sampling
                        const width = this.size[0];
                        const height = 10;
                        const y = 45;
                        
                        // Draw background
                        ctx.fillStyle = "#333";
                        ctx.fillRect(10, y, width - 20, height);
                        
                        // Draw frame markers
                        ctx.fillStyle = "#6E6";
                        const spacing = (width - 20) / Math.min(frameCount, 20);
                        for (let i = 0; i < Math.min(frameCount, 20); i++) {
                            ctx.fillRect(10 + i * spacing, y, 2, height);
                        }
                        
                        // Show which frames are sampled
                        const samplesWidget = this.widgets.find(w => w.name === "frame_sample_count");
                        if (samplesWidget) {
                            const sampleCount = samplesWidget.value;
                            ctx.fillStyle = "#F66";
                            if (frameCount > sampleCount) {
                                for (let i = 0; i < sampleCount; i++) {
                                    const idx = Math.floor(i * (frameCount-1) / (sampleCount-1));
                                    const x = 10 + (idx / (frameCount-1)) * (width - 20);
                                    ctx.fillRect(x-2, y-2, 4, height+4);
                                }
                            }
                        }
                    }
                    
                    // Draw the model name
                    const modelWidget = this.widgets.find(w => w.name === "model_name");
                    if (modelWidget) {
                        ctx.fillStyle = "#AAA";
                        ctx.font = "12px Arial";
                        ctx.fillText(`Model: ${modelWidget.value}`, 10, 70);
                    }
                }
            };
            
            // Update the canvas when widget values change
            const origOnWidgetChange = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function(widget, name, oldValue, newValue) {
                if (origOnWidgetChange) {
                    origOnWidgetChange.apply(this, arguments);
                }
                
                if (name === "frame_sample_count" || name === "analysis_type" || name === "fps" || 
                    name === "model_name" || name === "input_mode") {
                    this.setDirtyCanvas(true, false);
                }
                
                // Auto-update dependencies based on selected options
                if (name === "input_mode") {
                    // Show/hide relevant widgets based on input mode
                    const isFrameMode = newValue === "Frames";
                    const videoWidget = this.widgets.find(w => w.name === "video_file");
                    const imagesWidget = this.widgets.find(w => w.name === "images");
                    
                    if (videoWidget) {
                        videoWidget.hidden = isFrameMode;
                    }
                    if (imagesWidget) {
                        imagesWidget.hidden = !isFrameMode;
                    }
                }
            };
            
            // Add info about execution time
            nodeType.prototype.onExecuted = function(message) {
                if (message && message.status === "executed") {
                    this.execution_time = message.execution_time || 0;
                    this.setDirtyCanvas(true, false);
                }
            };
            
            // Add custom right-click menu options
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                options.push({
                    content: "📝 Analysis Options",
                    callback: () => {
                        const fpsWidget = this.widgets.find(w => w.name === "fps");
                        const sampleCountWidget = this.widgets.find(w => w.name === "frame_sample_count");
                        const maxPixelsWidget = this.widgets.find(w => w.name === "max_pixels");
                        
                        const msg = `
                        <div style="background-color: #2a2a2a; padding: 10px; border-radius: 5px;">
                            <h3>Video Analysis Settings</h3>
                            <p><b>FPS:</b> ${fpsWidget ? fpsWidget.value : "N/A"} - Controls frame sampling rate</p>
                            <p><b>Frame Samples:</b> ${sampleCountWidget ? sampleCountWidget.value : "N/A"} - Number of frames to analyze</p>
                            <p><b>Max Pixels:</b> ${maxPixelsWidget ? maxPixelsWidget.value : "N/A"} - Image resolution limit</p>
                            <hr/>
                            <p><i>Higher FPS (8-16) gives better motion analysis but uses more memory</i></p>
                            <p><i>More frame samples (12-24) improves analysis quality but increases processing time</i></p>
                        </div>
                        `;
                        LiteGraph.alert(msg);
                    }
                });
                
                options.push({
                    content: "🎬 Optimize for video type",
                    has_submenu: true,
                    callback: (value, options, e, menu, node) => {
                        const preset_options = [
                            "Short clip (5-10s)",
                            "Medium video (10-30s)",
                            "Long video (30s+)",
                            "Action scene",
                            "Dialog scene",
                            "High FPS footage"
                        ];
                        
                        new LiteGraph.ContextMenu(preset_options, {
                            event: e,
                            callback: (preset) => {
                                const fpsWidget = this.widgets.find(w => w.name === "fps");
                                const sampleCountWidget = this.widgets.find(w => w.name === "frame_sample_count");
                                const maxPixelsWidget = this.widgets.find(w => w.name === "max_pixels");
                                
                                switch(preset) {
                                    case "Short clip (5-10s)":
                                        fpsWidget.value = 8.0;
                                        sampleCountWidget.value = 16;
                                        maxPixelsWidget.value = 512*384;
                                        break;
                                    case "Medium video (10-30s)":
                                        fpsWidget.value = 4.0;
                                        sampleCountWidget.value = 20;
                                        maxPixelsWidget.value = 448*336;
                                        break;
                                    case "Long video (30s+)":
                                        fpsWidget.value = 2.0;
                                        sampleCountWidget.value = 24;
                                        maxPixelsWidget.value = 384*288;
                                        break;
                                    case "Action scene":
                                        fpsWidget.value = 12.0;
                                        sampleCountWidget.value = 24;
                                        maxPixelsWidget.value = 512*384;
                                        break;
                                    case "Dialog scene":
                                        fpsWidget.value = 4.0;
                                        sampleCountWidget.value = 12;
                                        maxPixelsWidget.value = 512*512;
                                        break;
                                    case "High FPS footage":
                                        fpsWidget.value = 16.0;
                                        sampleCountWidget.value = 32;
                                        maxPixelsWidget.value = 448*336;
                                        break;
                                }
                                
                                for (const widget of this.widgets) {
                                    widget.callback?.(widget.value);
                                }
                                
                                this.setDirtyCanvas(true, false);
                            },
                            parentMenu: menu,
                            node: node
                        });
                    }
                });
            };
            
            // Add hover tooltip
            nodeType.prototype.onNodeHovered = function(ctx) {
                if (!this.flags.collapsed) {
                    const mousePos = app.canvas.last_mouse;
                    
                    ctx.fillStyle = "rgba(50, 50, 50, 0.7)";
                    ctx.fillRect(mousePos.x, mousePos.y - 40, 240, 35);
                    ctx.fillStyle = "#FFF";
                    ctx.font = "12px Arial";
                    
                    const inputMode = this.widgets.find(w => w.name === "input_mode")?.value || "Frames";
                    if (inputMode === "Frames") {
                        ctx.fillText("Frame Mode: Connect image sequence from LoadVideo", mousePos.x + 5, mousePos.y - 20);
                    } else {
                        ctx.fillText("File Mode: Select video file from input directory", mousePos.x + 5, mousePos.y - 20);
                    }
                }
            };
        }
    },
    
    async nodeCreated(node) {
        if (node.comfyClass === "VideoPromptNode") {
            // Set default values and styling for newly created nodes
            const frameSampleWidget = node.widgets.find(w => w.name === "frame_sample_count");
            if (frameSampleWidget && frameSampleWidget.value < 16) {
                frameSampleWidget.value = 16;
            }
            
            const fpsWidget = node.widgets.find(w => w.name === "fps");
            if (fpsWidget && fpsWidget.value < 8.0) {
                fpsWidget.value = 8.0;
            }
            
            // Set node color and size
            node.color = "#4a6d8c";
            node.bgcolor = "#2a3d4c";
            
            // Set minimum size
            node.size = node.computeSize();
            if (node.size[0] < 300) node.size[0] = 300;
            if (node.size[1] < 120) node.size[1] = 120;
        }
    }
});