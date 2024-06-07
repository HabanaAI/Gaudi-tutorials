import gradio as gr


def simple_chat_gui(generate_fn):
    gr.set_static_paths(paths=["gui/assets/"])
    
    theme = gr.themes.Base(
        primary_hue=gr.themes.Color(
            c100="#dbeafe", c200="#bfdbfe", c300="#93c5fd", c400="#60a5fa", c50="#eff6ff", c500="#0054ae", c600="#00377c", c700="#00377c", c800="#1e40af", c900="#1e3a8a", c950="#0a0c2b"),
        secondary_hue=gr.themes.Color(
            c100="#dbeafe", c200="#bfdbfe", c300="#93c5fd", c400="#60a5fa", c50="#eff6ff", c500="#0054ae", c600="#0054ae", c700="#0054ae", c800="#1e40af", c900="#1e3a8a", c950="#1d3660"),
    ).set(
        body_background_fill_dark='*primary_950',
        body_text_color_dark='*neutral_300',
        border_color_accent='*primary_700',
        border_color_accent_dark='*neutral_800',
        block_background_fill_dark='*primary_950',
        block_border_width='2px',
        block_border_width_dark='2px',
        button_primary_background_fill_dark='*primary_500',
        button_primary_border_color_dark='*primary_500'
    )
    
    css='''
        @font-face {
        font-family: 'IntelOne';
        src: url('file=gui/assets/intelone-text-regular.woff') format('woff'); 
    }  
    
    .banner {
        display: flex;
        border-radius: 25px;
    	height: 100px;
        background: #0068b5;
      	justify-content: space-between !important;
      	flex-wrap: nowrap !important;
      	align-items: center !important;
      	position: relative ;
    }
    
    
    .banner h2 {
    	color: white;
    	margin: 0;
      	position: absolute;
      	left: 50%;
      	transform: translate(-50%);
      	top: 35%;
        text-align: center;
        font-family: 'IntelOne';
        font-weight: 300;
        font-style: normal;
    }
    
    .banner img {
    	width: 70px;
      	height: 70px;
        margin: 4px;
    }
    
    .badges {
        display: flex;
        margin-left: 60px;
        
    }
    #component-6 {height: 300px !important}
    .user-row {width: 70%; align-self: flex-end}
    .bot-row {width: 70% ; align-self: flex-start}
    
    .upload-button {
        pointer-events: none;
        visibility: hidden;
    
    }

    .show-api {
        pointer-events: none;
        visibility: hidden;
    }
    .built-with {
        pointer-events: none;
        visibility: hidden;
    }
    
    .gradio-container { font-family: IntelOne }
    
    '''
    #.gradio-container { font-family: IntelOne }
    html_banner = '''
    <div class="banner">
    <div class="badges">
    <img src="file/gui/assets/gaudi-badge-3000.png">
    </div>
    <div class="title">
    <h2 >RAG on Intel® Gaudi® 2 AI Accelerator</h2>
    </div>
    
    </div>
    '''
    
    with gr.Blocks(analytics_enabled=False, theme=theme, css=css) as demo:
        with gr.Row():
            gr.HTML(value=html_banner)
        with gr.Row():
            with gr.Column(scale=4):
                gr.ChatInterface(
                    fn=generate_fn,
                    examples=[{"text": "what is the summary of this document ?"}, {"text": "When was the 2024 Indian General elections held ?"}],
                    theme=theme,
                    css=css,
                    multimodal=True,
                )
            #with gr.Column(scale=1):
            #    gr.Markdown(f" WIP: space for vectorDB file upload")
            #    gr.File()
    return demo