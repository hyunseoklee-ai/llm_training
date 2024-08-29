from transformers import AutoModelForCausalLM, AutoTokenizer
# from accelerate import Accelerator
model_name = 'google/gemma-2-2B'

tokenizer = AutoTokenizer.from_pretrained(model_name)


# accelerator = Accelerator(project_dir="./test/gemma")
# device = accelerator.device
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model.state_dict()

# model = accelerator.prepare_model(model)
# accelerator.save_state(output_dir="./test/gemma")
