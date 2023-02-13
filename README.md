# Seam carving

An implementation of seam carving in CUDA.

Removing multiple seam:

1. Calculate grayscale
2. Calculate energy map
3. Calculate seam 
4. Remove seam from original image and grayscale 
5. Return to step 
