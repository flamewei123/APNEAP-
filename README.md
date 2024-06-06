# APNEAP-
Codes for paper "Mitigating Privacy Seesaw in Large Language Models: Augmented Privacy Neuron Editing via Activation Patching"

## Abstract

Protecting privacy leakage in large language models remains a paramount challenge. 
In this paper, we reveal Privacy Seesaw in LLM privacy protection via neuron editing, a phenomenon where measures to secure specific private information inadvertently heighten exposure risks for other privacy. 
Through comprehensive analysis, we identify the amount of targeted privacy data and the volume of edited privacy neurons as the two central triggers to this issue. 
To mitigate privacy seesaw, we propose Augmented Privacy Neuron Editing via Activation Patching (APNEAP), a novel framework designed to well balance model performance with privacy protection. 
The proposed APNEAP augments collected private data by automatically synthesizing new private data, which deactivates the first trigger to the privacy seesaw issue. 
Additionally, it adapts activation patching to privacy neuron editing for switching off the second trigger to the privacy seesaw problem. 
Experimental results show that the proposed APNEAP is capable of alleviating the privacy seesaw phenomenon and offers a more stable and reliable approach to privacy protection in LLMs than previous methods. 

## Overview
![image](https://github.com/flamewei123/APNEAP-/blob/main/overview-APNEAP.png)
