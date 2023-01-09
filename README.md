# Seam carving

Final project in HCMUS parallel programming course. An implementation of seam carving in CUDA.

Tasks: [Link](https://docs.google.com/spreadsheets/d/1niNkgiatqVv5ZHbLDha_LVsxlsbOHn0Jkyx4X0cTt-Q/edit?usp=sharing)

Notebook: [Link](https://colab.research.google.com/drive/1dG64BS-6aJgMvjVhSdZmtZ1NyFAy_ts9?usp=sharing)

Use `host_answer_gen` to generate answers first before testing.

Removing multiple seam:

1. Calculate grayscale
2. Calculate energy map
3. Calculate seam 
4. Remove seam from original image and grayscale 
5. Return to step 
