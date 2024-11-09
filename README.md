# aind-smartspim-external-dispatcher

Repository that hosts the code, environment and metadata of the Code Ocean capsule used
to dispatch parallel image processing steps in the SmartSPIM pipeline using the Code Ocean pipeline feature.

This is necessary since we need to make sure we process SmartSPIM channels in parallel. In order to do this, we rely on the flatten connection between the results folder generated in this capsule and other capsules such as [aind-smartspim-segmentation](https://github.com/AllenNeuralDynamics/aind-SmartSPIM-segmentation), [aind-ccf-registration](https://github.com/AllenNeuralDynamics/aind-ccf-registration) and [aind-smartspim-quantification](https://github.com/AllenNeuralDynamics/aind-smartspim-quantification).

Capsule modes:
- "dispatch": This mode dispatches multiple instances of the downstream capsules by using the flatten connection. It should be used in the middle of the pipeline.

- "clean": This mode cleans up all the results from the downstream capsules because our data is being copied to the aind-open-data bucket. It should be used at the end of the pipeline.