# barnacle_counter
How to use:

First, you'll need to have OpenCV, pyTorch, numpy, and matplotlib libraries downloaded on your computer. Then you'll have to have the images saved to your computer and in the same file as functions.py and main-barnacle_counter.py. Finally, you'll need to load SAM to your device (I recommend using a GPU or cloud GPU since the model is inherently large). Do this by pip installing segment-anything and git+https://github.com/facebookresearch/segment-anything.git. Then, specifically grabbing the smaller vit_b model by using !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth. 

Note that I have provided an .ipynb to show the whole process of loading the model and all libraries, plus showing the final model working with a final output. You can use this file directly, but I recommend using the provided functions.py and main-barnacle_counter.py files directly, which do the same thing in a cleaner format.  

Overview:

This is a Python program that implements the Metas Segment Anything Model to count the number of Barnacles in an unseen image of barnacles. The program utilizes helper functions first to resize an image. Then, another program was used to calculate a range of segment pixel sizes in which a valid barnacle should lie. This function is a low level of "training" the model. Using the data returned by this function, we can calculate the minimum and maximum of the range. We then use a function to segment our unseen images, store the valid segments by comparing their sizes to our found range. Once stored, we can count the number of valid segments in the list and return the count. Furthermore, we can access specific features of each segment object in this list to plot the location of each potential barnacle and plot each with a dot proportional to its segment pixel size. 

Issues and improvements: 

Currently, the model runs well on one of the unseen images. The image that runs well is the unseen image with a more zoomed-in and cropped image of the barnacles. I think the issue with the larger image is that the barnacles are so small, and when we resize the image to be able to run on a low-memory GPU or CPU, the image has such a low resolution that the barnacles blur into one large mass in which the segmentation model can't pick up as multiple barnacles. This, in turn, leads our model to output only a fraction of the true barnacle count.

In the future, to remedy this issue, we can use the OpenCV library to automate a cropping function that takes a large image, crops the important parts of the image, all while maintaining high resolution. This fix seems reasonable with more proficiency with the OpenCV library, but I was unable to automate the process with my current skill level with the library. 

Another smaller, yet still important, issue is the segment anything model. The model works well, but not great, as even in the image with a cropped and zoomed-in picture of barnacles, we can still see the model undercounts the barnacles by 10-20 barnacles. I guess that the model needs more "training data" for our specific problem. I'm not aware of a true training method for the Segment Anything Model since it is so large. But having more pre-masked images with a correct count and a training function could allow us to fine-tune our model.  

Key Takeaways:

Before this project, I had never implemented the use of a large pre-trained model. It was very exciting to use such an amazing, complex, and well-developed tool. This let me focus more on the problem at hand, which was fitting this tool to the specific problem. Furthermore, being able to use this general model and fit it to a specific task unlocks so many more future projects in which identifying objects in an image or video is required. 
