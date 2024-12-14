from math import sqrt
import torch
import torch.nn as nn
import os

from transformers import LlamaConfig, LlamaModel, MistralModel, AutoTokenizer,  \
    BitsAndBytesConfig, MistralConfig
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

from modules.MultiscaleDecomp import MultiscaleDecomp
from modules.MultiscaleMix import MultiscaleMix
from modules.GraphExtractor.Attention import AttEncoderLayer
from modules.GraphExtractor.VGAE import VGAE

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

transformers.logging.set_verbosity_error()

class OutProjection(nn.Module):
    def __init__(self, llm_dim, d_model, patch_nums, Ti_list, down_sampling_layers):
        super().__init__()
        self.channel_compression = nn.Linear(llm_dim, d_model)
        self.flatten = nn.Flatten(start_dim=-2)
        self.out_layers = torch.nn.ModuleList(
            [
                nn.Linear(
                        patch_nums[i]*d_model,
                        Ti_list[i]
                    )
                    for i in range(down_sampling_layers + 1)
            ]
        )
        
    def forward(self, dec_out, patch_num_list):
        # [B, N, (L_0+L_1+...+L_multiscale), 4096]
        dec_out = self.channel_compression(dec_out) # [B, N, (L_0+L_1+...+L_multiscale), d_model]
        split_sizes = [T_i for T_i in patch_num_list]
        dec_out_list = torch.split(dec_out, split_sizes, dim=2) # [multiscale, B, N, L_i, d_model]
        multiscale_list = []
        for i, dec_out in enumerate(dec_out_list):
            dec_out = self.flatten(dec_out) # [B, N, L_i*d_model]
            dec_out = self.out_layers[i](dec_out) # [B, N, L_i*d_model]->[B, N, T_i]
            multiscale_list.append(dec_out.permute(0, 2, 1)) # [B, T_i, N]
        return multiscale_list # [multiscale, B, T_i, N]

class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.d_ff = args.d_ff
        self.top_k = 5
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.ts_len = args.ts_len
        self.device = args.device
        self.down_sampling_layers = args.down_sampling_layers
        self.llm_model_name = args.llm_model

        self.multiscaleDecomp = MultiscaleDecomp(args)
        self.multiscaleMix = MultiscaleMix(args)

        path = args.llm_path

        if args.llm_model == 'LLAMA3':
            print('model:LLAMA3')
            self.llama_config = LlamaConfig.from_pretrained(path)
            self.llama_config.num_hidden_layers = args.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                llm_model = LlamaModel.from_pretrained(
                    path,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    load_in_4bit=True,
                    device_map='auto',
                    torch_dtype=torch.bfloat16
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                llm_model = LlamaModel.from_pretrained(
                    path,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config
                )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    path,
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif args.llm_model == 'Mistral':
            print('model:Mistral')
            self.mistral_config = MistralConfig.from_pretrained(path)
            self.mistral_config.num_hidden_layers = args.llm_layers
            self.mistral_config.output_attentions = True
            self.mistral_config.output_hidden_states = True

            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            try:
                llm_model = MistralModel.from_pretrained(
                    path,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.mistral_config,
                    quantization_config=nf4_config, 
                    use_safetensors=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                llm_model = MistralModel.from_pretrained(
                    path,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.mistral_config,
                    quantization_config=nf4_config, 
                    use_safetensors=True
                )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    path,
                    trust_remote_code=True,
                    local_files_only=True,
                    use_safetensors=True,
                    quantization_config=nf4_config
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    path,
                    trust_remote_code=True,
                    local_files_only=False,
                    use_safetensors=True,
                    quantization_config=nf4_config
                )
        else:
            raise Exception('LLM model is not defined')
        peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=args.r, lora_alpha=args.lora_alpha, \
            lora_dropout=args.lora_dropout, # r:lora attention dimension(the rank)
            target_modules=["q_proj", "k_proj"])
        self.llm_model = get_peft_model(llm_model, peft_config)
        self.llm_model.print_trainable_parameters()

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
            
        self.llm_model = self.llm_model.eval()
        self.description = args.content
        self.adj_text = args.adj_text
        
        self.dropout_param = args.dropout
        self.dropout = nn.Dropout(args.dropout)

        self.embedding_layer = nn.Conv2d(in_channels=1, out_channels=args.d_model, kernel_size=1)
        self.patch_embedding = PatchEmbedding(
            args.d_model, self.patch_len, self.stride, args.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens).to(self.device)

        self.reprogramming_layer = ReprogrammingLayer(self.device, args.d_model, args.n_heads, self.d_ff, args.llm_dim)
        
        self.patch_nums = [(args.ts_len//(args.down_sampling_window**i)+self.stride-self.patch_len)//self.stride+1
                           for i in range(args.down_sampling_layers + 1)] # patch_nums[i]=L_i
        self.Ti_list = [args.ts_len//(args.down_sampling_window**i) for i in range(args.down_sampling_layers + 1)]
        self.out_projection = OutProjection(args.llm_dim, args.d_model, self.patch_nums, self.Ti_list, args.down_sampling_layers)
        self.normalize_layers = Normalize(args.nodes_num, affine=False)

        self.VGAE = VGAE(args, args.adj)
        self.att = AttEncoderLayer(args, self.ts_len, args.att_hidden, args.n_head, args.d_v, args.d_k, args.dropout)

    def forward(self, x_input):
        # [B, T, N]
        x_input = self.normalize_layers(x_input, 'norm') # (x-mu)/std
        
        B, T, N = x_input.size()

        # ---------get prompt based on data information-----------
        prompt = []
        prompt_ = (
        f"Dataset description: {self.description}"
        f"Prior knowledge: {self.adj_text}"
        )

        if self.llm_model_name == 'LLAMA3':
            messages = [
                {"role": "system", "content": "You are a data analysis expert. Your task is to help users analyze fMRI time series data to extract brain effective connectivity network. "},
                {"role": "user", "content": prompt_}
            ]
        elif self.llm_model_name == 'Mistral':
            settings = "You are a data analysis expert. Your task is to help users analyze fMRI time series data to extract brain effective connectivity network."
            messages = [
                {"role": "user", "content": settings + prompt_},
            ]
        
        # ---------prompt tokenizer and embedding---------
        prompt = self.tokenizer.apply_chat_template(
		messages, 
		tokenize=True, 
		add_generation_prompt=True,
        )
        prompt = torch.tensor(prompt)
        prompt = prompt.unsqueeze(0)
        prompt = prompt.repeat(B * N, 1) # [B*N, L_p]

        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(self.device))  # (B*N, prompt_token, 4096), that is [B*N, L_p, 4096]
        # [128256, 4096]->[4096, 128256]->[4096, 1000](num_tokens=1000)->[1000, 4096]
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0) # Input to the linear layer

        # Multiscale Decomposition Mixing
        multiscale_list = self.multiscaleDecomp(x_input) # [multiscale, B, T_i, N]

        # reprogrammed patch embedding
        enc_out_list = []
        patch_num_list = []
        for scale in multiscale_list:
            enc_out = scale.permute(0, 2, 1).contiguous() # [B, N, T_i]
            enc_out, _ = self.patch_embedding(enc_out.to(torch.bfloat16)) # [B*N, (T+stride-patch_len)/stride+1, d_model]
            # ****** def L_i = (T_i+stride-patch_len)/stride+1
            enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings) # [B*N, L_i, 4096]
            
            L_i = enc_out.shape[1]
            patch_num_list.append(L_i)
            enc_out_list.append(enc_out) 
        # enc_out_list:[multiscale, B*N, L_i, 4096]

        concat_multiscale = torch.cat(enc_out_list, dim=1) # [B*N, (L_0+L_1+...+L_multiscale), 4096]
        llama_enc_out = torch.cat([prompt_embeddings, concat_multiscale], dim=1) # [B*N, L_p+(L_0+L_1+...+L_multiscale), 4096]

        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state # [B*N, L_p+(L_0+L_1+...+L_multiscale), 4096]
        
        # Multiscale Reconstruction Mixing
        dec_out = dec_out[:, prompt_embeddings.shape[1]:, :] # [B*N, (L_0+L_1+...+L_multiscale), 4096]
        dec_out = dec_out.view(B, N, concat_multiscale.shape[1], concat_multiscale.shape[2])
        multiscale_list = self.out_projection(dec_out, patch_num_list) # [B, N, (L_0+L_1+...+L_multiscale), 4096]->[multiscale, B, T_i, N]

        # Mixing multiple scales
        L_constrastive, multiscale_mix = self.multiscaleMix(multiscale_list) # value, [B, T, N]

        # VGAE_out, adj = self.VGAE(multiscale_mix) # [B, T, N]
        # adj = torch.mean(adj, dim=0)
        att_out, adj = self.att(multiscale_mix)
        dec_out = self.normalize_layers(att_out, 'denorm') # x*std+mu
        # dec_out = self.normalize_layers(VGAE_out, 'denorm') # x*std+mu

        return L_constrastive, dec_out, adj

    def save_pretrained(self, dir, params):
        dir = dir + str(self.args.data) + '/sim' + str(self.args.index) + '/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        params_with_grad = {name: param for name, param in self.named_parameters() if param.requires_grad}
        save_path = os.path.join(dir, params + '.pth')
        torch.save(params_with_grad, save_path)

    def load_pretrained(self, dir, params):
        dir = dir + str(self.args.src_data) + '/sim' + str(self.args.src_index) + '/'
        load_path = os.path.join(dir, params + '.pth')

        if os.path.exists(load_path):
            saved_params = torch.load(load_path)
            current_params = {name: param for name, param in self.named_parameters() if param.requires_grad}

            # 加载保存的参数到模型
            current_params.update(saved_params)
            self.load_state_dict(current_params, strict=False)
        else:
            raise FileNotFoundError(f"No file found at {load_path}")

    def calc_subjects_diff(self, adj):
        # [B, N, N]
        adj_expanded_1 = adj.unsqueeze(1)  # [B, 1, N, N]
        adj_expanded_2 = adj.unsqueeze(0)  # [1, B, N, N]
        diff = adj_expanded_1 - adj_expanded_2  # [B, B, N, N]
        diff_squared = diff ** 2
        mse = diff_squared.mean(dim=[2, 3]) # [B, B]
        return mse.sum(dim=[0, 1])/2 # a value

    def calcute_lags(self, x_input):
        q_fft = torch.fft.rfft(x_input.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_input.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, device, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads).to(device)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads).to(device)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads).to(device)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm).to(device)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        # [B*N, (T+stride-patch_len)/stride+1, d_model],[1000, 4096],[1000, 4096]
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
    # attention operation
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        # torch.einsum(): Sums the product of the elements of the input operands along dimensions specified using a notation based on the Einstein summation convention.
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding