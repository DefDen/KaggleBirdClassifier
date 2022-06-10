## I Was Busy Thinkin' 'Bout Birds

<iframe width="560" height="315" src="https://www.youtube.com/embed/EJ0AqL_HTA4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Goals

My goal was to get the highest accuracy I could in the [Kaggle bird classifying competition](https://www.kaggle.com/competitions/birds22sp/). To achieve this I tested multiple combinations of neural networks with different optimization schemes and hyperparameters.

### Methodology

I started with the example code given by in class [here](https://colab.research.google.com/drive/1kHo8VT-onDxbtS3FM77VImG35h_K_Lav?usp=sharing), which served as a baseline for future experiments. The default neural network is based off resnet18 and runs the SGD optimizer over 5 epochs with 128x128 images, which yields a testing accuracy of 49%. 

Following advice in class, I decided to mainly focus my work on finding the best starting neural network, optimizer and image augmentations.

### resnet18 and resnet152

![resnet](https://user-images.githubusercontent.com/31528205/172783119-92cfa108-80da-4c2b-832f-b1f5836ef15d.png)

After toying around with resnet18, I decided to use a larger neural network to hopefully achieve better results. I chose resnet152 since it was the largest resnet according the [documentation](https://pytorch.org/hub/pytorch_vision_resnet/) (in my head at the time bigger = better). While this was true in terms of accuracy, (with identical hyperparameters resnet152 achieved an accuracy of 59% to resnet18's 49%), it also cost 10 times more time to train. On my laptop with an 11th Gen Intel Core i7 CPU it took 1 hour to train resnet18, while it took 11 to train resnet152. At this point I chose to stick with resnet152, a decision I would very much regret later. 

### SGD and Adam

After doing more research into [optimizers]([https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6](https://ruder.io/optimizing-gradient-descent/)) it seemed like adaptive moment estimation or Adam was a popular choice for its speed and accuracy. However I was unable to get Adam to converge on an acceptable loss, even after running for 50 epochs it converged on a loss of 5.5 yielding a disappointing accuracy of 0.2% (this little "experiment" took 60 hours to run on my laptop by the way, and as we get into the next session is the cause of me changing to a different resnet). Overall if given more time, I think this is were I would try to improve my cnn the most. I am not sure why I couldn't get Adam to work, and it can potentially speed up by training by a considerable amount.

### Kaggle and resnet50

After the infamous training of resnet152 over 50 epochs I realized that I had not been giving speed enough priority. So far I had been training on my laptop with no gpu, so I first switched over to Kaggle. Kaggle provides gpus to train on which sped up our process tremendously. I also decided to change from resnet152 to resnet50. After testing, I found that with identical hyperparameters resnet50 trained 3 to 4 times faster than resnet152, and with a minimal hit to accuracy of 54% compared to 59%. 

### Additional data augmentation

In addition to the current data augmentation we have, I added [random perspective](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomPerspective.html), which randomly changes the perspective of an image. I doubt that this will mess with the data like a horizontal flip would, and it provides more safeguard to overfitting. In addition, I took the advice to use 256x256 images instead of 128x128 images. The extra fidelity in each photo helps our network, as there is more 4 times the information to work with.

### Preprocessing images

An issue that kept popping up while running the cnn in Kaggle was that the gpu kept alternating between 0 and 100% utilization. The results were a very slow training session, sometimes even slower than my laptop. After a while of troubleshooting, I realized that it might be related to [resizing the images](https://www.kaggle.com/product-feedback/165588). By now, I have been testing inputting 256x256 images instead of the default 128x128, and I assumed that the cpu doing the resizing was the bottleneck. To fix this I made a new dataset bird256 with all the images already downsized to 256x256 with a shell script, which seems to have largely fixed the issue. Now the cpu is under less strain and data is given to the gpu faster, reducing the bottleneck.

### Results

After learning these larger lessons and a lot more fine tinkering with the learning rate and number of epochs trained, I landed on a final accuracy of 73.5%. I used resnet50 with a learning rate of 0.01 for 20 epochs, followed by continuing training for another 10 epochs at a learning rate of 0.001. I found in testing that running at a higher learning rate for the full 30 epochs caused overfitting, but running at a lower learning rate for more epochs often took too long or didn't achieve the accuracy I was looking for.

