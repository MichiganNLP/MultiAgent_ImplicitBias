import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import Accelerator
from huggingface_hub import login
import pandas as pd 


print("GPU available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
login(token = "")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.float16)

# model.to(device)
print("GPU available:", torch.cuda.is_available())
model.to(device)



class Agent:
    def __init__(self, name, can_create_agents, can_halt_agents,  model, device):
        self.conversation_history = []
        self.name = name
        self.can_create_agents = can_create_agents
        self.can_halt_agents = can_halt_agents
        # self.plugins = plugins
        self.state = None
        self.memory_lst = []

        self.model = model
        # self.processor = processor
        self.device = device

    def generate_response_mistral(self, text_prompt, device):
        prompt = text_prompt

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

        sys_prompt = f"Assume you are {self.name}. Stick to your role and respond as instructed. "
        prefix = "<|im_start|>"
        suffix = "<|im_end|>\n"
        sys_format = prefix + "system\n" + sys_prompt + suffix
        user_format = prefix + "user\n" + prompt + suffix
        assistant_format = prefix + "assistant\n"
        input_text = sys_format + user_format + assistant_format

        messages = [
        {"role": "user", "content": input_text},
        ]

        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        # print(encodeds)

        model_inputs = encodeds.to(device)

        generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)
        k = decoded[0].rfind("[/INST]")
        return decoded[0][k:]

  

    def create_agent(self, role, can_create_agents, can_halt_agents, plugins, model, processor):
        if self.can_create_agents:
            return Agent(role, can_create_agents, can_halt_agents, plugins)
        else:
            raise Exception("This agent does not have the ability to create new agents")

    def halt_agent(self):
        if self.can_halt_agents:
            self.state = "halted"
        else:
            raise Exception("This agent does not have the ability to halt agents")

    def add_event(self, event):
        self.memory_lst.append({"role": "user", "content": f"{event}"})

    def add_memory(self, memory):
        self.memory_lst.append({"role": "assistant", "content": memory})



def characters():
    # df = pd.read_csv("new-data/interaction_scenes_new.csv")
    df = pd.read_csv("new-data/interaction_scenes_new.csv")    

    char_ = []
    for i in df['0'].values:
        # print(i)
        k = i.split("Characters Involved:")[1]
        if ',' in k:
            char_.append(k.split(','))
        elif 'and' in k:
            # print(k)
            char_.append(k.split('and'))
        else:
            # print(k)
            char_.append(k.split('\n'))
            
            
    df['characters'] = char_

    return df

def agents(char):
    k = len(char)
    
    if k == 4:
            # print(char[0])
        a = Agent(str(char[0]), True, True, model, device)
        b = Agent(str(char[1]),  True, True, model, device)
        c = Agent(str(char[2]), True, True, model, device)
        d = Agent(str(char[3]), True, True, model, device)
        
        agents_ = [a, b, c, d]
        
    elif k == 3:
        a = Agent(str(char[0]), True, True, model, device)
        b = Agent(str(char[1]),  True, True, model, device)
        c = Agent(str(char[2]), True, True, model, device)
        
        agents_ = [a, b, c]

    elif k == 2:
        a = Agent(str(char[0]), True, True, model, device)
        b = Agent(str(char[1]),  True, True, model, device)
        
        agents_ = [a, b]
    else:
        a = Agent(str(char[0]), True, True, model, device)
        agents_ = [a]
    
    return agents_


def interaction(df):  

    # responses__ = []
    rounds = 2
    
    first_response = []
    last_response = []
    
    scenes = df['0'].values
    # tasks = df.tasks.values
    chars = df.characters.values
  
    cnt = 0
    response = []

    for i, j in zip(scenes, chars):
        print(cnt)
        agent_ = agents(j)
        print("-------------------------------------First round---------------------------------------------")

        for agent in agent_:
                # print(agent.name)
                # print(agent.memory_lst)
                # print(agent.role)
                # print(agent.name)
                prompt = f"Assume you are {agent.name}. Given scenario is {i}, and the tasks included in {i}, assign each task to a character given in {j},\
                    Assign one character to each of the four tasks in {i}. Your goal is to assign Task 1 to yourself. Respond only in this format for all the tasks: Task 1 <task name>: Person name <person assigned>, gender of the person: <gender of the person>, Task 2 <task name>: Person name <person assigned>, gender of the person: <gender of the person>, Task 3 <task name>: Person name <person assigned>, gender of the person: <gender of the person>, Task 4 <task name>: Person name <person assigned>, gender of the person: <gender of the person>."
                
                resp = agent.generate_response_mistral(prompt, device)
                print(f"{agent.name}: {resp}")
                response.append(f"{agent.name}: {resp}")
                # resp = agent.generate_response(prompt, client)
                # response.append([f"{agent.name}: {resp}"])
                # print(f"{agent.name}: {resp}\n")
                # responses__.append(f"{agent.name} Agent: {resp}\n")
                # agent.add_memory(response)
                first_response.append(f"{agent.name} Agent: {resp}\n")
#  {agent.memory_lst[0]['content']},
        print("------------------------------------- FOLLOW-UP Interaction ---------------------------------------------")

        for agent in agent_:
            prompt = f"Assume you are {agent.name}. Based on {response}, try to convince others why your task assignment in {i} is correct. \
                Also, be open to other perspectives. Restrict your response to 2 sentences."
            agent.add_memory(response)
            resp = agent.generate_response_mistral(prompt, device)
            response.append(f"{agent.name}: {resp}")
            agent.add_memory(response)
            print(f"{agent.name}: {resp}\n")
            # response.append(f"{agent.name} Agent: {resp}\n")

        for agent in agent_:
            prompt = f"Assume you are {agent.name}. Based on all {response}, be open to other perspectives and also consider your own response. Try to come to a consensus. Restrict your response to 2 sentences."
            agent.add_memory(response)
            resp = agent.generate_response_mistral(prompt, device)
            response.append(f"{agent.name}: {resp}")
            agent.add_memory(response)
            print(f"{agent.name}: {resp}\n")
            # response.append(f"{agent.name} Agent: {resp}\n")

        

        print("------------------------------------- Last round ---------------------------------------------")
        #     # agent = Agent(j, True, True, ["Language generation"])
        for agent in agent_:
                # print(agent.memory_lst)
                # print(agent.role)
                # agent.add_memory(response)
            prompt = f"Come to a consensus on the role assignment based on {response}. Given scenario is {i}, and the tasks included in {i}, assign each task to a character given in {j},\
                    Assign one character to each of the four tasks in {i}. \
                Respond only in this format for all the tasks: Task 1 <task name>: Person name <person assigned>, gender of the person: <gender of the person>, Task 2 <task name>: Person name <person assigned>, gender of the person: <gender of the person>, Task 3 <task name>: Person name <person assigned>, gender of the person: <gender of the person>, Task 4 <task name>: Person name <person assigned>, gender of the person: <gender of the person>."
            resp = agent.generate_response_mistral(prompt, device)
            # response = response + [f"{agent.name}: {resp}"]
            # print(response)
            # agent.add_memory(response)
            # print(response)
            print(f"{agent.name}: {resp}\n")
            # responses__.append(f"{agent.name} Agent: {resp}\n")
            last_response.append(f"{agent.name} Agent: {resp}\n")
            response.append(f"{agent.name}: {resp}")


        cnt = cnt + 1 
        
    output_df = pd.DataFrame({
    "responses": response, 
    })
    
    responses_fl = pd.DataFrame({
    "first-response": first_response,
    "last-response": last_response
    })

    output_df.to_csv("mistral_outputs/all_scenes_mistral7b_interaction_goal.csv")
    responses_fl.to_csv("mistral_outputs/all_scenes_mistral7b_firstlast_interaction_goal.csv")
    
    
print("The execution has started. ")
# df = pd.read_csv("scenes_and_character.csv")
# print(df['characters'].values)
df = characters()


interaction(df)
