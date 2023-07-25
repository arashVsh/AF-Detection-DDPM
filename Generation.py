
from DDPM import MyDDPM
from UNET import MyUNet
from Settings import *
from EPS_Saver import eps_saveImage
# import torch
# from torchviz import make_dot
# X = torch.randn(1, 100)

# model = MyUNet()
# y = model(X)

# make_dot(y.mean(), params=dict(model.named_parameters()))

def generate(samples, folderName:str):
    from npy_loader import custom_loader
    from ImageResizer import cropper, magnifier
    import random
    import numpy as np
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from torch.optim import Adam

    LOSS_FILE_NAME = 'Loss.txt'
    if folderName == 'eps_images/1': # First call of generate function
        with open(LOSS_FILE_NAME, 'w') as file:
            file.truncate(0) # Clear the content of LOSS_FILE_NAME file

    # Setting reproducibility
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    """ Execution options """

    no_train = False
    store_path = "ddpm.pt"

    samples = cropper(samples, imageNewSize=IMAGE_SIZE)
    loader = custom_loader(samples)

    """ Utility functions """

    def show_images(images, title):
        """Shows the provided images as sub-pictures in a square"""

        # Converting images to CPU numpy arrays
        if type(images) is torch.Tensor:
            images = images.detach().cpu().numpy()

        # Defining number of rows and columns
        fig = plt.figure(figsize=(8, 8))
        rows = min(int(len(images) ** (1 / 2)), 5)
        cols = min(round(len(images) / rows), 5)

        # Populating figure with sub-plots
        idx = 0
        for r in range(rows):
            for c in range(cols):
                fig.add_subplot(rows, cols, idx + 1)

                if idx < len(images):
                    plt.imshow(images[idx][0], cmap="gray")
                    idx += 1
        fig.suptitle(title, fontsize=30)

        # Showing the figure
        plt.show()
        eps_saveImage(images[0, :, :], folderName=folderName + '/Noising/', fileName=title)


    def show_first_batch(loader):
        for batch in loader:
            show_images(batch[0], "Images in the first batch")
            break

    # Optionally, show a batch of regular images
    if showFirstBatch:
        show_first_batch(loader)

    """## Getting device

    If you are running this codebook from Google Colab, make sure you are using a GPU runtime. For non-pro users, typically a *Tesla T4* GPU is provided.
    """

    # Getting device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using device: {device}\t"
        + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU")
    )

    def show_forward(ddpm, loader, device):
        # Showing the forward process
        for batch in loader:
            imgs = batch[0]
            # show_images(imgs, "Original images")
            for percent in [0.25, 0.5, 0.75, 1]:
                show_images(
                    ddpm(
                        imgs.to(device),
                        [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))],
                    ),
                    f"DDPM Noisy images {int(percent * 100)}%",
                )

            break

    def generate_new_images(
        ddpm,
        n_samples,
        device=torch.device("cuda"),
        c=1,
    ):
        
        """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""

        with torch.no_grad():
            if device is None:
                device = ddpm.device

            # Starting from random noise
            x = torch.randn(n_samples, c, IMAGE_SIZE, IMAGE_SIZE).to(device)

            for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
                # Estimating noise to be removed
                time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
                eta_theta = ddpm.backward(x, time_tensor)

                alpha_t = ddpm.alphas[t]
                alpha_t_bar = ddpm.alpha_bars[t]

                # Partially denoising the image
                x = (1 / alpha_t.sqrt()) * (
                    x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
                )

                if t > 0:
                    z = torch.randn(n_samples, c, IMAGE_SIZE, IMAGE_SIZE).to(device)
                    beta_t = ddpm.betas[t]
                    if firstNoisingMethod:
                        sigma_t = beta_t.sqrt()
                    else:
                        prev_alpha_t_bar = (
                            ddpm.alpha_bars[t - 1] if t > 0 else ddpm.alphas[0]
                        )
                        beta_tilda_t = (
                            (1 - prev_alpha_t_bar) / (1 - alpha_t_bar)
                        ) * beta_t
                        sigma_t = beta_tilda_t.sqrt()

                    # Adding some more noise like in Langevin Dynamics fashion
                    x = x + sigma_t * z
        return x

    """# Instantiating the model """

    # Defining model
    ddpm = MyDDPM(
        MyUNet(n_steps),
        image_chw=(1, IMAGE_SIZE, IMAGE_SIZE),
        n_steps=n_steps,
        min_beta=min_beta,
        max_beta=max_beta,
        device=device,
    )

    sum([p.numel() for p in ddpm.parameters()])

    """# Optional visualizations"""

    # Optionally, load a pre-trained model that will be further trained
    # ddpm.load_state_dict(torch.load(store_path, map_location=device))

    global showForward
    if showForward:
        showForward = False
        show_forward(ddpm, loader, device)

    """ Training loop """

    def training_loop(
        ddpm, loader, n_epochs, optim, device, display, store_path
    ):
        mse = nn.MSELoss()
        best_loss = float("inf")
        n_steps = ddpm.n_steps

        for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
            epoch_loss = 0.0
            for step, batch in enumerate(
                tqdm(
                    loader,
                    leave=False,
                    desc=f"Epoch {epoch + 1}/{n_epochs}",
                    colour="#005500",
                )
            ):
                # Loading data
                x0 = batch[0].to(device)
                n = len(x0)

                # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
                eta = torch.randn_like(x0).to(device)
                t = torch.randint(0, n_steps, (n,)).to(device)

                # Computing the noisy image based on x0 and the time-step (forward process)
                noisy_imgs = ddpm(x0, t, eta)

                # Getting model estimation of noise based on the images and the time-step
                eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

                # Optimizing the MSE between the noise plugged and the predicted noise
                loss = mse(eta_theta, eta)
                optim.zero_grad()
                loss.backward()
                optim.step()

                epoch_loss += loss.item() * len(x0) / len(loader.dataset)

            log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"
            # if display and epoch % 25 == 0:
            #     show_images(
            #         generate_new_images(ddpm, n_samples=25, device=device),
            #         f"Images generated at epoch {epoch + 1}",
            #     )

            # Storing the model
            if best_loss > epoch_loss:
                best_loss = epoch_loss
                torch.save(ddpm.state_dict(), store_path)
                log_string += " --> Best model ever (stored)"

                # Display images generated at this epoch
                # if display and epoch >= 199:
                #     show_images(
                #         generate_new_images(ddpm, n_samples=25, device=device),
                #         f"Images generated at epoch {epoch + 1}",
                #     )

            print(log_string)
            with open(LOSS_FILE_NAME, "a") as file:
                file.write(log_string + "\n")

        with open(LOSS_FILE_NAME, "a") as file:
            file.write("\n*************************************************************************************************************\********************\n\n")

    # Training
    if not no_train:
        training_loop(
            ddpm,
            loader,
            N_EPOCHS,
            optim=Adam(ddpm.parameters(), LEARNING_RATE),
            device=device,
            display=False,
            store_path=store_path,
        )

    """ Testing the trained model """

    # Loading the trained model
    best_model = MyDDPM(
        MyUNet(), image_chw=(1, IMAGE_SIZE, IMAGE_SIZE), n_steps=n_steps, device=device
    )
    best_model.load_state_dict(torch.load(store_path, map_location=device))
    best_model.eval()
    print("Model loaded")

    print("Generating new images...")
    generated = generate_new_images(best_model, n_samples=N_FAKE_SAMPLES % 100)
    print("generated.shape: ", generated.shape)
    show_images(generated, "Final result")
    generated = generated.cpu().numpy()

    for i in range(N_FAKE_SAMPLES // 100):
        generated = np.concatenate(
            (generated, generate_new_images(best_model, n_samples=100).cpu().numpy()),
            axis=0,
        )

        print("generated.shape: ", generated.shape)

    generated = np.squeeze(generated, axis=1)
    generated = magnifier(generated)
    for i in range(15):
        eps_saveImage(generated[i, :, :], folderName=folderName + '/Generated_Images/', fileName=str(i))
    return generated
