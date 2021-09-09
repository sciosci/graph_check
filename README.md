# Graphical integrity issues in open access publications: detection and patterns of proportional ink violations
A deep learning-based provess to measure violations of the proportional ink principle. The specific rules are: a bar chart’s y-axis should start from zero, have one scale, and not be partially hidden (Bergstrom & West, 2020; Tufte, 2001). The AUC of the method is 0.917 with 0.02 standard deviation and 0.77 precision with 0.0209 based on 5-fold cross-validation. 

# Requirement
1. YoloV4 (Bochkovskiy et al., 2020)
```bash
git clone https://github.com/AlexeyAB/darknet.git
```
2. Reverse-Engineering Visualizations(REV) (Poco & Heer, 2017)
```bash
git clone https://github.com/uwdata/rev.git
```
    - Move REV.py inside the rev.git folder before doing REV. Follow the instruction in the notebook to implement.
3. Stroke Width Transform (Epshtein et al., 2010)
   - We did some adjustment of stroke-width-transform. (https://github.com/sunsided/stroke-width-transform.git) 
   - Use convert_swt.py to do the transformation.

# Method 
## Method FlowChart
<img src="https://github.com/PeterHuang024/Graphical_Integrity_Issues/blob/main/images/flowchart.png" alt="drawing" width="1200"/>

# Example
![Image](https://github.com/PeterHuang024/Graphical_Integrity_Issues/blob/main/images/Example1.png) | ![Image](https://github.com/PeterHuang024/Graphical_Integrity_Issues/blob/main/images/Example2.png)
:-------------------------:|:-------------------------:
![Image](https://github.com/PeterHuang024/Graphical_Integrity_Issues/blob/main/images/Example3.png) | ![Image](https://github.com/PeterHuang024/Graphical_Integrity_Issues/blob/main/images/Example4.png)
The y-axis of upper two graphs does not start from zero and there are truncations in lower two graphs. Therefore, these graphs would be annotated graphical integrity issues.

# Reference
- Bergstrom, C. T., & West, J. D. (2020). Calling Bullshit: The Art of Skepticism in a Data-Driven World (Illustrated Edition). Random House.
- Tufte, E. R. (2001). The visual display of quantitative information (Vol. 2). Graphics press Cheshire, CT.
- Bochkovskiy, A., Wang, C.-Y., & Liao, H.-Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. ArXiv:2004.10934 [Cs, Eess]. http://arxiv.org/abs/2004.10934
- Poco, J., & Heer, J. (2017). Reverse-engineering visualizations: Recovering visual encodings from chart images. Computer Graphics Forum, 36, 353–363.
- Epshtein, B., Ofek, E., & Wexler, Y. (2010). Detecting text in natural scenes with stroke width transform. 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2963–2970. https://doi.org/10.1109/CVPR.2010.5540041
