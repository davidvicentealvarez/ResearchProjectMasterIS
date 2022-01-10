#%%

import os
import numpy as np
import torch
from data import ceasar_shift, convert_data
from torch import nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.modelV2 import GeneratorV2, DiscriminatorV2
from models.ResNet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.model import Generator, Discriminator
from models.cyphergan_models import GeneratorV3, DiscriminatorV3, DiscriminatorV4, GeneratorV4
import matplotlib.pyplot as plt
import wandb
import datetime
from torch.autograd import Variable
from torch import autograd
from tqdm import tqdm

#%% md

## Init parameters

#%%

os.environ["WANDB_SILENT"] = "true"
# os.environ["WANDB_MODE"] = "offline"
torch.manual_seed(1)
np.random.seed(1)
# init parameters
wandb.config = {
    "version" : 3,
    "batch_size" : 64,
    "train_split" : 0.9,
    "test_split": 0.95,
    "num_epochs" : 50,
    "lr_gen" : 0.0002,
    "lr_discr" : 0.0002,
    "lr_embedding" : 0.0002,
    "beta1" : 0.5,
    "beta2" : 0.999,
    "device" :  "cuda:1" if torch.cuda.is_available() else "cpu",
    "shift" : 10,
    "reg" : 1,
    "instance_size" : 100,
    "dictionary_size" : 27,
    "discriminator_step" : 1,
    "generator_step" : 1,
    "lambda_term":10
}
run = wandb.init(project="Research_project_IS", entity="davidvicente", name=str(datetime.datetime.now()), config=wandb.config)

#%% md

## Load data


#%%

## Create data
np_data = convert_data(fixed_len=wandb.config["instance_size"])
np_crypted_data = ceasar_shift(np_data, wandb.config["shift"])

tensor_clear_text = torch.from_numpy(np_data)
tensor_crypted_data = torch.from_numpy(np_crypted_data)

tensor_clear_text = tensor_clear_text.float().view(-1, 1, wandb.config["instance_size"], wandb.config["dictionary_size"])
tensor_crypted_data = tensor_crypted_data.float().view(-1, 1, wandb.config["instance_size"], wandb.config["dictionary_size"])

#%% md

## Shuffle Data, and split it into train/test/validation splits (60/20/20)

#%%

num_train = len(tensor_clear_text)
indices_clear = list(range(num_train))
indices_crypted = list(range(num_train))
np.random.shuffle(indices_clear)
np.random.shuffle(indices_crypted)
train_split = int(np.floor(wandb.config["train_split"] * num_train))
test_split = int(np.floor(wandb.config["test_split"] * num_train))


clear_txt_train = tensor_clear_text[indices_clear[:train_split]]
clear_txt_test = tensor_clear_text[indices_clear[train_split:test_split]]
clear_txt_valid = tensor_clear_text[indices_clear[test_split:]]

crypted_txt_train = tensor_crypted_data[indices_crypted[:train_split]]
crypted_txt_test = tensor_crypted_data[indices_crypted[train_split:test_split]]
crypted_txt_valid = tensor_crypted_data[indices_crypted[test_split:]]

#%% md

## Create data loaders

#%%

train_clear_loader = DataLoader(clear_txt_train, batch_size=wandb.config["batch_size"])
train_crypted_loader = DataLoader(crypted_txt_train, batch_size=wandb.config["batch_size"])

#%% md

## Init generators and discriminators

#%%

if wandb.config["version"] ==  1 :
    crypted_gen = Generator(wandb.config["instance_size"], wandb.config["dictionary_size"]).to(wandb.config["device"]) # converts clear to crypted
    clear_gen = Generator(wandb.config["instance_size"], wandb.config["dictionary_size"]).to(wandb.config["device"]) # converts crypted to clear
    crypted_discr = Discriminator(wandb.config["instance_size"], wandb.config["dictionary_size"]).to(wandb.config["device"])
    clear_discr = Discriminator(wandb.config["instance_size"], wandb.config["dictionary_size"]).to(wandb.config["device"])
elif wandb.config["version"] ==  2:
    crypted_gen = GeneratorV2(wandb.config["instance_size"], wandb.config["dictionary_size"]).to(wandb.config["device"]) # converts clear to crypted
    clear_gen = GeneratorV2(wandb.config["instance_size"], wandb.config["dictionary_size"]).to(wandb.config["device"]) # converts crypted to clear
    crypted_discr = DiscriminatorV2(wandb.config["instance_size"], wandb.config["dictionary_size"]).to(wandb.config["device"])
    clear_discr = DiscriminatorV2(wandb.config["instance_size"], wandb.config["dictionary_size"]).to(wandb.config["device"])
elif wandb.config["version"] ==  3:
    crypted_gen = GeneratorV3(wandb.config["instance_size"], wandb.config["dictionary_size"]).to(wandb.config["device"]) # converts clear to crypted
    clear_gen = GeneratorV3(wandb.config["instance_size"], wandb.config["dictionary_size"]).to(wandb.config["device"])# converts crypted to clear
    crypted_discr = DiscriminatorV3(wandb.config["instance_size"], wandb.config["dictionary_size"]).to(wandb.config["device"])
    clear_discr = DiscriminatorV3(wandb.config["instance_size"], wandb.config["dictionary_size"]).to(wandb.config["device"])
elif wandb.config["version"] ==  4:
    crypted_gen = GeneratorV4(wandb.config["instance_size"], wandb.config["dictionary_size"]).to(wandb.config["device"]) # converts clear to crypted
    clear_gen = GeneratorV4(wandb.config["instance_size"], wandb.config["dictionary_size"]).to(wandb.config["device"])# converts crypted to clear
    crypted_discr = DiscriminatorV4(wandb.config["instance_size"], wandb.config["dictionary_size"]).to(wandb.config["device"])
    clear_discr = DiscriminatorV4(wandb.config["instance_size"], wandb.config["dictionary_size"]).to(wandb.config["device"])
else :
    crypted_gen = resnet101(1,wandb.config["instance_size"], wandb.config["dictionary_size"]).to(wandb.config["device"]) # converts clear to crypted
    clear_gen = resnet101(1,wandb.config["instance_size"], wandb.config["dictionary_size"]).to(wandb.config["device"]) # converts crypted to clear
    crypted_discr = resnet34(1,1).to(wandb.config["device"])
    clear_discr = resnet34(1,1).to(wandb.config["device"])

embedding = nn.Linear( wandb.config["dictionary_size"], wandb.config["dictionary_size"]).to(wandb.config["device"])

wandb.watch(crypted_gen, log="all", log_freq=1000, log_graph=True, idx=1)
wandb.watch(clear_gen, log="all", log_freq=1000, log_graph=True, idx=2)
wandb.watch(crypted_discr, log="all", log_freq=1000, log_graph=True, idx=3)
wandb.watch(clear_discr, log="all", log_freq=1000, log_graph=True, idx=4)
wandb.watch(embedding,log="all", log_freq=1000, log_graph=True, idx=5)

#%% md

## Init Optimizers and losses

#%%

# Setup Adam optimizers for both generators
optimizer_crypted_gen = optim.Adam(crypted_gen.parameters(), lr=wandb.config["lr_gen"], betas=(wandb.config["beta1"], wandb.config["beta2"]))
optimizer_clear_gen = optim.Adam(clear_gen.parameters(), lr=wandb.config["lr_gen"], betas=(wandb.config["beta1"], wandb.config["beta2"]))

# Setup Adam optimizers for both discriminators
optimizer_crypted_discr= optim.Adam(crypted_discr.parameters(), lr=wandb.config["lr_discr"], betas=(wandb.config["beta1"], wandb.config["beta2"]))
optimizer_clear_discr = optim.Adam(clear_discr.parameters(), lr=wandb.config["lr_discr"], betas=(wandb.config["beta1"], wandb.config["beta2"]))

optimizer_embedding = optim.Adam(embedding.parameters(), lr=wandb.config["lr_embedding"], betas=(wandb.config["beta1"], wandb.config["beta2"]))

# Create Losses
BCE = nn.BCELoss()
cross_entropy = nn.CrossEntropyLoss()
mse = nn.MSELoss()

#%%

# checkpoint = torch.load("checkpoint2.pt")
#
# crypted_gen.load_state_dict(checkpoint["crypted_gen_sate_dict"])
# clear_gen.load_state_dict(checkpoint["clear_gen_sate_dict"])
# crypted_discr.load_state_dict(checkpoint["crypted_discr_sate_dict"])
# clear_discr.load_state_dict(checkpoint["clear_discr_sate_dict"])
#
# optimizer_crypted_gen.load_state_dict(checkpoint["crypted_gen_optimizer"])
# optimizer_clear_gen.load_state_dict(checkpoint["clear_gen_optimizer"])
# optimizer_crypted_discr.load_state_dict(checkpoint["crypted_discr_optimizer"])
# optimizer_clear_discr.load_state_dict(checkpoint["clear_discr_optimizer"])
#
# epoch_start = checkpoint["epoch"]
#
# del checkpoint

#%% md

## Precompute test ground truth

#%%

with torch.no_grad():
    # Create the True encryption of each test instance
    test_crypt_np = crypted_txt_test.view(-1,wandb.config["instance_size"],wandb.config["dictionary_size"]).detach().numpy()
    test_decrypted_np = ceasar_shift(test_crypt_np, -wandb.config["shift"])
    test_decrypted_np_char = np.argmax(test_decrypted_np, axis=2)
    test_decrypted = torch.from_numpy(test_decrypted_np_char)
    crypted_txt_test = crypted_txt_test.float()

    # Create the True decryption of each test instance
    test_clear_np = clear_txt_test.view(-1,wandb.config["instance_size"],wandb.config["dictionary_size"]).detach().numpy()
    test_encrypted_np = ceasar_shift(test_clear_np, wandb.config["shift"])
    test_encrypted_np_char = np.argmax(test_encrypted_np, axis=2)
    test_encrypted = torch.from_numpy(test_encrypted_np_char)
    clear_txt_test = clear_txt_test.float()

#%% md

# Gradient penalty for discriminators

#%%

def compute_gradient_penalty(discriminator,embedding, real_data, fake_data):
    eta = torch.FloatTensor(real_data.size(0),1,1,1).uniform_(0,1).to(wandb.config["device"])
    eta = eta.expand(real_data.size(0), real_data.size(1), real_data.size(2), real_data.size(3))
    interpolated = eta * real_data + ((1 - eta) * fake_data).to(wandb.config["device"])
    interpolated = Variable(interpolated, requires_grad=True)
    prob_interpolated = discriminator(embedding(interpolated))
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).to(wandb.config["device"]),
                               create_graph=True, retain_graph=True)[0]
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * wandb.config["lambda_term"]
    return grad_penalty

#%% md

# Train loop

#%%

torch.autograd.set_detect_anomaly(True)
for epoch in tqdm(range(wandb.config["num_epochs"])):

    dataloader_iterator = iter(train_crypted_loader)

    for i, clear in enumerate(tqdm(train_clear_loader)):
        wandb.log({"epoch": epoch}, commit=False)

        crypted_gen.train()
        clear_gen.train()
        crypted_discr.train()
        clear_discr.train()
        embedding.train()

        try:
            crypted = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(train_clear_loader)
            crypted = next(dataloader_iterator)

        real_clear_text = clear.to(wandb.config["device"])
        real_crypted_text = crypted.to(wandb.config["device"])

        optimizer_embedding.zero_grad()

        if i%wandb.config["discriminator_step"] == 0:

            ###############################
            ### precompute clear fakes ####
            ###############################

            fake_clear_text = clear_gen(embedding(real_crypted_text), logits=False)

            ############################################
            ### empty clear discriminator gradients ####
            ############################################

            clear_discr.zero_grad()

            ###############################################
            ### train clear discriminator on true data ####
            ###############################################

            pred_real_clear = clear_discr(embedding(real_clear_text))
            # true_labels = torch.full((len(real_clear_text),1), 1, dtype=torch.float, device=wandb.config["device"])
            # batch_clear_d_true_loss = BCE(pred_real_clear, true_labels)
            batch_clear_d_true_loss = -torch.mean(pred_real_clear)
            wandb.log({"clear discr real error" : -pred_real_clear.mean().item()}, commit=False)
            # batch_clear_d_true_loss.backward()

            ###############################################
            ### train clear discriminator on fake data ####
            ###############################################

            pred_fake_clear  = clear_discr(embedding(fake_clear_text.detach()))
            # fake_labels = torch.full((len(real_clear_text),1), 0, dtype=torch.float, device=wandb.config["device"])
            # batch_clear_d_fake_loss = BCE(pred_fake_clear, fake_labels)
            batch_clear_d_fake_loss = torch.mean(pred_fake_clear)
            wandb.log({"clear discr fake error": pred_fake_clear.mean().item()} , commit=False)
            # batch_clear_d_fake_loss.backward()

            #########################################################
            ### Compute Gradient penalty for clear discriminator ####
            #########################################################

            gradient_penalty = compute_gradient_penalty(clear_discr, embedding, real_clear_text, fake_clear_text)
            wandb.log({"clear discr gradient penalty": gradient_penalty.item()}, commit=False)
            # gradient_penalty.backward()

            ################################################
            ### Compute final error clear discriminator ####
            ################################################

            error_d_clear = batch_clear_d_true_loss  + batch_clear_d_fake_loss  + gradient_penalty
            wandb.log({"loss discriminator clear": error_d_clear.item()}, commit=False)

            ################################################
            ### Optimize and update clear discriminator ####
            ################################################

            error_d_clear.backward()
            optimizer_clear_discr.step()

            #---------------------------------------#
            #---------------------------------------#

            #################################
            ### precompute crypted fakes ####
            #################################

            fake_crypted_text =  crypted_gen(embedding(real_clear_text), logits=False)

            ##############################################
            ### empty crypted discriminator gradients ####
            ##############################################

            crypted_discr.zero_grad()

            #################################################
            ### train crypted discriminator on true data ####
            #################################################

            pred_real_crypted = crypted_discr(embedding(real_crypted_text))
            # true_labels = torch.full((len(real_crypted_text),1), 1, dtype=torch.float, device=wandb.config["device"])
            # batch_crypted_d_true_loss = BCE(pred_real_crypted, true_labels)
            batch_crypted_d_true_loss = -torch.mean(pred_real_crypted)
            wandb.log({"crypted discr real error": -pred_real_crypted.mean().item()}, commit=False)
            # batch_crypted_d_true_loss.backward()

            #################################################
            ### train crypted discriminator on fake data ####
            #################################################

            pred_fake_crypted  = crypted_discr(embedding(fake_crypted_text.detach()))
            # fake_labels = torch.full((len(fake_crypted_text),1), 0, dtype=torch.float, device=wandb.config["device"])
            # batch_crypted_d_fake_loss = BCE(pred_fake_crypted, fake_labels)
            batch_crypted_d_fake_loss = torch.mean(pred_fake_crypted)
            wandb.log({"crypted discr fake error": pred_fake_crypted.mean().item()}, commit=False)
            # batch_crypted_d_fake_loss.backward()

            ###########################################################
            ### Compute Gradient penalty for crypted discriminator ####
            ###########################################################

            gradient_penalty = compute_gradient_penalty(crypted_discr, embedding, real_crypted_text, fake_crypted_text)
            wandb.log({"crypted discr gradient penalty": gradient_penalty.item()}, commit=False)
            # gradient_penalty.backward()

            ##################################################
            ### Compute final error crypted discriminator ####
            ##################################################

            error_d_crypted = batch_crypted_d_true_loss + batch_crypted_d_fake_loss + gradient_penalty
            wandb.log({"loss discriminator crypted": error_d_crypted.item()}, commit=False)

            ##################################################
            ### Optimize and update crypted discriminator ####
            ##################################################

            error_d_crypted.backward()
            optimizer_crypted_discr.step()

            #---------------------------------------#
            #---------------------------------------#

        if i%wandb.config["generator_step"] == 0:

            ###################################
            ### empty generators gradients ####
            ###################################

            clear_gen.zero_grad()
            crypted_gen.zero_grad()

            ##################################
            ### First reconstruction loss ####
            ##################################

            fake_crypted_reconstruct = crypted_gen(embedding(clear_gen(embedding(real_crypted_text), logits=False)), logits=True)
            # fake_crypted_reconstruct_loss = torch.sum(torch.square(fake_crypted_reconstruct - real_crypted_text), dim=(2,3)).mean() * wandb.config["reg"]
            # fake_crypted_reconstruct_loss = torch.linalg.norm((fake_crypted_reconstruct - real_crypted_text)**2, dim=(2,3)).mean() * wandb.config["reg"]
            fake_crypted_reconstruct_loss = cross_entropy(fake_crypted_reconstruct.view(-1, wandb.config["instance_size"], wandb.config["dictionary_size"]).transpose(1,2), torch.argmax(real_crypted_text, 3).view(-1,wandb.config["instance_size"])) * wandb.config["reg"]
            # fake_crypted_reconstruct_loss.backward()
            wandb.log({"crypted text reconstruction" : fake_crypted_reconstruct_loss.item()}, commit=False)

            ##################################
            ### Second reconstruction loss ###
            ##################################

            fake_clear_reconstruct =  clear_gen(embedding(crypted_gen(embedding(real_clear_text), logits=False)), logits=True)
            # fake_clear_reconstruct_loss = torch.sum(torch.square(fake_clear_reconstruct - real_clear_text), dim=(2,3)).mean() * wandb.config["reg"]
            # fake_clear_reconstruct_loss = torch.linalg.norm((fake_clear_reconstruct - real_clear_text)**2, dim=(2,3)).mean() * wandb.config["reg"]
            fake_clear_reconstruct_loss = cross_entropy(fake_clear_reconstruct.view(-1, wandb.config["instance_size"], wandb.config["dictionary_size"]).transpose(1,2), torch.argmax(real_clear_text, 3).view(-1,wandb.config["instance_size"])) * wandb.config["reg"]
            # fake_clear_reconstruct_loss.backward()
            wandb.log({"clear text reconstruction" : fake_clear_reconstruct_loss.item()}, commit=False)

            reconstruction_loss = fake_clear_reconstruct_loss + fake_crypted_reconstruct_loss
            reconstruction_loss.backward()
            optimizer_embedding.step()
            #################################################
            ### train clear generator with discriminator ####
            #################################################

            fake_clear_text =  clear_gen(embedding(real_crypted_text), logits=False)
            gen_fake_clear = clear_discr(embedding(fake_clear_text))
            # gen_labels = torch.full((len(fake_clear_text),1), 1, dtype=torch.float, device=wandb.config["device"])
            # fake_gen_clear_loss = BCE(gen_fake_clear, gen_labels)
            fake_gen_clear_loss = -torch.mean(gen_fake_clear)
            wandb.log({"clear gen error" : -gen_fake_clear.mean().item()}, commit=False)
            fake_gen_clear_loss.backward()



            ############################################
            ### Compute final error clear generator ####
            ############################################

            batch_clear_gen_loss = fake_gen_clear_loss + reconstruction_loss
            wandb.log({"loss generator clear": batch_clear_gen_loss.item()}, commit=False)

            ############################################
            ### Optimize and update clear generator ####
            ############################################

            # batch_clear_gen_loss.backward()
            optimizer_clear_gen.step()

            #---------------------------------------#
            #---------------------------------------#


            ###################################################
            ### train crypted generator with discriminator ####
            ###################################################

            fake_crypted_text = crypted_gen(embedding(real_clear_text), logits=False)
            gen_fake_crypted = crypted_discr(embedding(fake_crypted_text))
            # gen_labels = torch.full((len(fake_crypted_text),1), 1, dtype=torch.float, device=wandb.config["device"])
            # fake_gen_crypted_loss = BCE(gen_fake_crypted, gen_labels)
            fake_gen_crypted_loss = -torch.mean(gen_fake_crypted)
            wandb.log({"crypted gen error" : - gen_fake_crypted.mean().item()}, commit=False)
            fake_gen_crypted_loss.backward()

            ##############################################
            ### Compute final error crypted generator ####
            ##############################################

            batch_crypted_gen_loss = fake_gen_crypted_loss + reconstruction_loss
            wandb.log({"loss generator crypted": batch_crypted_gen_loss.item()}, commit=False)

            #############################################
            ### Optimize and update crypted generator ###
            #############################################

            # batch_crypted_gen_loss.backward()
            optimizer_crypted_gen.step()

            #---------------------------------------#
            #---------------------------------------#

            #########################################
            ### Compute final error of embedding ####
            #########################################
            embedding_loss = error_d_crypted + error_d_clear + reconstruction_loss
            wandb.log({"loss embedding": embedding_loss.item()}, commit=False)


            #---------------------------------------#
            #---------------------------------------#

        ##########################################
        ### Test performance of our generators ###
        ##########################################
        crypted_gen.eval()
        clear_gen.eval()
        crypted_discr.eval()
        clear_discr.eval()
        embedding.eval()

        with torch.no_grad():
            ########################################################
            ### Test performance of clear generator (decrypting) ###
            ########################################################

            test_decrypted_gen = clear_gen(embedding(crypted_txt_test.to(wandb.config["device"]))).detach().to("cpu").view(-1,wandb.config["instance_size"],wandb.config["dictionary_size"])
            test_decrypted_gen_char = torch.argmax(test_decrypted_gen, 2).view(-1)
            test_decrypted_accuracy = (test_decrypted.view(-1)==test_decrypted_gen_char).sum().item()/len(test_decrypted.view(-1))
            wandb.log({"test decrypting accuracy": test_decrypted_accuracy}, commit=False)

            ##########################################################
            ### Test performance of crypted generator (encrypting) ###
            ##########################################################

            test_encrypted_gen = crypted_gen(embedding(clear_txt_test.to(wandb.config["device"]))).detach().to("cpu").view(-1,wandb.config["instance_size"],wandb.config["dictionary_size"])
            test_encrypted_gen_char = torch.argmax(test_encrypted_gen, 2).view(-1)
            test_encrypted_accuracy = (test_encrypted.view(-1)==test_encrypted_gen_char).sum().item()/len(test_encrypted.view(-1))
            wandb.log({"test encrypting accuracy": test_encrypted_accuracy}, commit=True)






#%% md

# Set all the models to evaluation mode to validate performanced of our generators

#%%

crypted_discr.eval()
clear_discr.eval()

#%%

crypted_gen.eval()
clear_gen.eval()

#%% md

# Validate performance of clear generator (decrypting)

#%%

# with torch.no_grad():
#     clear_gen = clear_gen.to("cpu")
#     val_crypt_np = crypted_txt_valid.view(-1,wandb.config["instance_size"],wandb.config["dictionary_size"]).detach().numpy()
#     decrypted_np = ceasar_shift(val_crypt_np, -wandb.config["shift"])
#     decrypted_gen = clear_gen(crypted_txt_valid).view(-1,wandb.config["instance_size"],wandb.config["dictionary_size"]).detach().numpy()
#     decrypted_np_char = np.argmax(decrypted_np, axis=2).reshape(-1)
#     decrypted_gen_char = np.argmax(decrypted_gen, axis=2).reshape(-1)
#     print((decrypted_np_char == decrypted_gen_char).mean())
#
# #%% md
#
# # Validate performance of crypted generator (encrypting)
#
# #%%
#
# with torch.no_grad():
#     embedding.to("cpu")
#     crypted_gen = crypted_gen.to("cpu")
#     val_clear_np = clear_txt_valid.view(-1,wandb.config["instance_size"],wandb.config["dictionary_size"]).detach().numpy()
#     encrypted_np = ceasar_shift(val_clear_np,wandb.config["shift"])
#     encrypted_gen = crypted_gen(clear_txt_valid).view(-1,wandb.config["instance_size"],wandb.config["dictionary_size"]).detach().numpy()
#     encrypted_np_char = np.argmax(encrypted_np, axis=2).reshape(-1)
#     encrypted_gen_char = np.argmax(encrypted_gen, axis=2).reshape(-1)
#     print((encrypted_np_char == encrypted_gen_char).mean())

#%% md

# Save models if necessary

#%%


torch.save({
    "crypted_gen_sate_dict" : crypted_gen.state_dict(),
    "clear_gen_sate_dict" : clear_gen.state_dict(),
    "crypted_discr_sate_dict" : crypted_discr.state_dict(),
    "clear_discr_sate_dict" : clear_discr.state_dict(),
    "embedding_state_dict" : embedding.state_dict(),
    "crypted_gen_optimizer" : optimizer_crypted_gen.state_dict(),
    "clear_gen_optimizer" : optimizer_clear_gen.state_dict(),
    "crypted_discr_optimizer" : optimizer_crypted_discr.state_dict(),
    "clear_discr_optimizer" : optimizer_clear_discr.state_dict(),
    "embedding_optimizer": optimizer_embedding.state_dict()
}, "checkpoint2.pt")

