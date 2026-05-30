<div align="center">

# Three-Body DFT Analysis

Using a Discrete Fourier Transform to analyse the motion in a three-body gravitational systems.

<div align="left">

<div align="center">

## Reflection on AI use
<div align="left">

The goal of the group was to minimize and if possible eliminate AI usage, especially in terms of creating logic, while still relying on it for creating GUI components, brainstorming and debugging of erroneous results.

### Which tools were used?
The AI tools used were: Claude Sonnet 4.6 & Google Gemini 3.5 Flash

### How and why were they used?
In the end, there were **[X]** scenarios for which AI was used, brainstorming, creating the GUI, answering questions regarding logic that we didn't fully grasp or that the other group member created,... . 

In terms of brainstorming, the group knew what area of interest was to be persued, that being orbital mechanics within a 3-body model. However, struggling to find use cases for DFT in the field Gemini was consulted. While the group knew that DFT could be used to find the orbital periods of quasi-periodic orbits, we weren't sure of the applications of knowing the orbital period. Prompts such as "What applications does knowing the orbital period have?" were used. While this didn't directly provide our final goal, it nonetheless helped to inspire us to **[INSERT GOAL HERE]**.

From the beginning the team was aware that creating the GUI was not the primary purpose of the project, and due to its menial nature it was decided that use of AI here would not be unfounded. Doing this we would provide the structure of existing logic methods as well as the task to the generative AI "Add a selector for using either Euler or Range-Kutta for integration" or "here is some psuedocode for how the GUI should work, could you implement it in python using PyQt6." The AI would then provide a code snippet that would be inserted into `GUI.py`.

Moreover, there were instances in which a group member made several commits involving logic, in which case the second person did on occasion use AI to explain the algorithmic process to them. Such was the case when the `DFTAnalysis.py` was pushed to the common repository by the first member. The second member then would provide the code along with a prompt such as "Explain what is happening, step by step, in the method DFTCharacteristicAnalysis." The AI would then provide a very useful breakdown of what is happening step by step, which helped get the second member up to speed.

### Describe at least one incorrect assumption, or misleading output and why an explanation of why it was incorrect or insufficient.

### Which tasks were the tools not used for?