# Graphical integrity issues in open access publications: detection and patterns of proportional ink violations
A deep learning-based provess to measure violations of the proportional ink principle. The specific rules are: a bar chart’s y-axis should start from zero, have one scale, and not be partially hidden (Bergstrom & West, 2020; Tufte, 2001). The AUC of the method is 0.917 with 0.02 standard deviation and 0.77 precision with 0.0209 based on 5-fold cross-validation. 

# Requirement
1. YoloV4 
```bash
git clone https://github.com/AlexeyAB/darknet.git
```
2. Reverse-Engineering Visualizations(REV)
```bash
git clone https://github.com/uwdata/rev.git
```

# Reference
- Bergstrom, C. T., & West, J. D. (2020). Calling Bullshit: The Art of Skepticism in a Data-Driven World (Illustrated Edition). Random House.
- Tufte, E. R. (2001). The visual display of quantitative information (Vol. 2). Graphics press Cheshire, CT.

