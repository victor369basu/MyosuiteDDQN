# Musculoskeletal Simulation with Double DQN reinforcement learning

<video width="400" height="400" playsinline autoplay muted loop src="./assets/myoSarcHandObjHoldRandom-v0-model.mp4">
</video>


https://user-images.githubusercontent.com/32032481/203810883-ce6abf5c-960b-4431-8f29-697e47ef5c18.mp4


Solved task - myoSarcHandObjHoldRandom-v0

# Introduction
The dexterous human hand requires the coordination of multiple highly redundant muscles, which have complementary and antagonistic effects on various joints. This musculoskeletal model is comprised of 29 bones, 23 joints, and 39 muscle-tendon units. Our objective is to simulate the coordination of these bones and muscles for different tasks via reinforcement learning.

In this repository, we try to solve musculoskeletal tasks with `Double DQN reinforcement learning`. A `transformer` model has been used as the base model architecture.

<img width="1240" alt="TasksALL" src="https://github.com/facebookresearch/myosuite/blob/main/docs/source/images/myoSuite_All.png?raw=true">

# Tools and Technologies
1. `MyoSuite` is a collection of musculoskeletal environments and tasks simulated with the [MuJoCo](http://www.mujoco.org/) physics engine and wrapped in the OpenAI ``gym`` API to enable the application of Machine Learning to bio-mechanic control problems.

2. `PyTorch` an open source machine learning framework that accelerates the path from research prototyping to production deployment.

# Double DQN
The standard DQN method has been shown to overestimate the true Q-value, because for the target an argmax over estimated Q-values is used. Therefore when some values are overestimated and some underestimated, the overestimated values have a higher probability to be selected.

**Standard DQN target:**  
Q(s<sub>t</sub>, a<sub>t</sub>) = r<sub>t</sub> + Q(s<sub>t+1</sub>, argmax<sub>a</sub>Q(s<sub>t</sub>, a))  

By using two uncorralated Q-Networks we can prevent this overestimation. In order to save computation time we do gradient updates only for one of the Q-Networks and periodically update the parameters of the target Q-Network to match the parameter of the Q-Network that is updated.  

**The Double DQN target then becomes:**  
Q(s<sub>t</sub>, a<sub>t</sub>) = r<sub>t</sub> + Q<sub>&theta;</sub>(s<sub>t+1</sub>, argmax<sub>a</sub>Q<sub>target</sub>(s<sub>t</sub>, a))  

**And the loss function is given by:**  
(Q(s<sub>t</sub>, a<sub>t</sub>) - Q<sub>&theta;</sub>(s<sub>t</sub>, a<sub>t</sub>))^2

# Non-stationarities task variations

## Sarcopenia
Sarcopenia is a muscle disorder that occurs commonly in the elderly population (Cruz-Jentoft and Sayer (2019)) and is characterized by a reduction in muscle mass or volume. The peak in grip strength can be reduced by up to 50% from age 20 to 40 (Dodds et al. (2016)). The simulation dataset modelled sarcopenia for each muscle as a reduction of 50% of its maximal isometric force.

## Fatigue
Muscle Fatigue is a short-term (second to minute) effect that happens after sustained or repetitive voluntary movement and it has been linked to traumas e.g. cumulative trauma disorder (Chaffin et al. (2006)). This model was based on the idea that different types of muscle fibre have different contributions and resistance to fatigue (Vøllestad (1997)). The current implementation is simplified to consider the same fatigue factor for all muscles and that muscle can be completely fatigued.

<img alt="Fatigue" src="https://raw.githubusercontent.com/facebookresearch/myosuite/main/docs/source/images/Fatigue.png">

## Tendon transfer
Contrary to muscle fatigue or sarcopenia that occurs in all muscles, tendon transfer surgery can target a single muscle-tendon unit. Tendon transfer surgery allows redirecting the application point of muscle forces from one joint DoF to another. It can be used to regain functional control of a joint or limb motion after injury. One of the current procedures in the hand is the tendon transfer of the Extensor Indicis Proprius (EIP) to replace the Extensor Pollicis Longus (EPL) (Gelb (1995)). Rupture of the EPL can happen after a broken wrist and create a loss of control of the Thumb extension. The MyoSuite module comes with a physical tendon transfer where the EIP application point of the tendon was moved from the index to the thumb and the EPL was removed.

<img alt="Fatigue" src="https://raw.githubusercontent.com/facebookresearch/myosuite/main/docs/source/images/tendon_transfer.png">

# Suite

MyoSuite consists of three models: :ref:`myoFinger`, :ref:`myoElbow` and :ref:`myoHand`. Using these models the MyoSuite module design a rich collection of tasks ranging from simple reaching movements to contact-rich movements like pen-twirling and baoding balls.

It also consists of three Musculoskeletal condition Variations: :ref:`sarcopenia`, :ref:`fatigue`, :ref:`ttransfer`
# Results

The left side video represents the simulation after training the transformer model with Double DQN reinforcement learning, and the right side video represents the simulation before training the transformer model with Double DQN reinforcement learning. Also, the transformer model and training hyper-parameters that have been used are mentioned w.r.t. each task.

## myoHandReachFixed-v0
```python
!python main.py --env_name myoHandReachFixed-v0 --gamma 0.99 --learning_rate 0.0003 
--eps 0.09 --batch 64 --epochs 200 --loss_fn mse --train True
```
<video playsinline autoplay muted loop src="./assets/myoHandReachFixed-v0.mp4"></video>


https://user-images.githubusercontent.com/32032481/203811052-bb2c6400-5091-4e80-8adc-209173e9310e.mp4


## myoHandReachRandom-v0
```python
!python main.py --env_name myoHandReachRandom-v0 --gamma 0.99 --learning_rate 0.00003 
--eps 0.09 --batch 32 --epochs 500 --loss_fn cel --train True
```
<video playsinline autoplay muted loop src="./assets/myoHandReachRandom-v0.mp4"></video>


https://user-images.githubusercontent.com/32032481/203811385-e05973b2-ec34-4d49-8161-298da29e9fa8.mp4


## myoSarcHandPose1Fixed-v0
```python
!python main.py --env_name myoSarcHandPose1Fixed-v0 --gamma 0.99 --learning_rate 0.00003
 --eps 0.09 --batch 64 --epochs 400 --loss_fn cel --train True
```
<video playsinline autoplay muted loop src="./assets/myoSarcHandPose1Fixed-v0.mp4">
</video>


https://user-images.githubusercontent.com/32032481/203811604-9d2bfaec-8b57-47a9-9c9a-6180140d55ec.mp4


## myoHandObjHoldFixed-v0
```python
!python main.py --env_name myoHandObjHoldFixed-v0 --gamma 0.99 --learning_rate 0.00003
 --eps 0.09 --batch 64 --epochs 400 --loss_fn cel --train True
```
<video playsinline autoplay muted loop src="./assets/myoHandObjHoldFixed-v0.mp4">
</video>


https://user-images.githubusercontent.com/32032481/203811935-cd0b2921-d184-4661-ae60-09c3d6b2060b.mp4


## myoSarcHandObjHoldRandom-v0
```python
!python main.py --env_name myoSarcHandObjHoldRandom-v0 --gamma 0.99 --learning_rate 0.0000003
 --eps 0.09 --batch 64 --epochs 600 --loss_fn cel --train True
```
<video playsinline autoplay muted loop src="./assets/myoSarcHandObjHoldRandom-v0.mp4">
</video>


https://user-images.githubusercontent.com/32032481/203812055-11247412-3821-475e-8426-88a1d16e8262.mp4


## myoHandKeyTurnFixed-v0
```python
!python main.py --env_name myoHandKeyTurnFixed-v0 --gamma 0.99 --learning_rate 0.00003
 --eps 0.09 --batch 64 --epochs 600 --loss_fn cel --train True
```
<video playsinline autoplay muted loop src="./assets/myoHandKeyTurnFixed-v0.mp4">
</video>


https://user-images.githubusercontent.com/32032481/203812160-c515c716-435d-4d1f-9f80-a5ec07a8ef7a.mp4


## myoSarcHandPenTwirlFixed-v0
```python
!python main.py --env_name myoSarcHandPenTwirlFixed-v0 --gamma 0.5 --learning_rate 0.000003
 --eps 0.09 --batch 64 --epochs 600 --loss_fn mse --train True
```
<video playsinline autoplay muted loop src="./assets/myoSarcHandPenTwirlFixed-v0.mp4">
</video>


https://user-images.githubusercontent.com/32032481/203812291-b5241d95-7e73-4c42-97c1-b85ecd4da94d.mp4


## myoChallengeDieReorientP1-v0
```python
!python main.py --env_name myoChallengeDieReorientP1-v0 --gamma 0.5 --learning_rate 0.000003
 --eps 0.09 --batch 64 --epochs 600 --loss_fn cel --train True
```
<video playsinline autoplay muted loop src="./assets/myoChallengeDieReorientP1-v0.mp4">
</video>


https://user-images.githubusercontent.com/32032481/203812402-c25c6d4e-3fe3-4e30-85e1-791cffadce01.mp4


## myoChallengeBaodingP1-v1
```python
!python main.py --env_name myoChallengeBaodingP1-v1 --gamma 0.6 --learning_rate 0.0000003
 --eps 0.09 --batch 64 --epochs 700 --loss_fn cel --train True
```
<video playsinline autoplay muted loop src="./assets/myoChallengeBaodingP1-v1.mp4">
</video>


https://user-images.githubusercontent.com/32032481/203812464-758f2746-ebbc-469b-888a-72f22ce2be4d.mp4


## myoFatiElbowPose1D6MExoRandom-v0
```python
!python main.py --env_name myoFatiElbowPose1D6MExoRandom-v0 --gamma 0.99 --learning_rate 0.0003
 --eps 0.09 --batch 64 --epochs 400 --loss_fn mse --train True
```
<video playsinline autoplay muted loop src="./assets/myoFatiElbowPose1D6MExoRandom-v0.mp4">
</video>


https://user-images.githubusercontent.com/32032481/203812530-5bade738-fc14-4b01-b75d-511f9bb2096e.mp4


# Inference
```python
!python main.py --env_name myoHandReachFixed-v0 --train False --model_save_path ./model/
```
The following code saves loads the trained model from the directory and runs the simulation, saving it to a video.

# Conclusion
* Transformer Model works well for most of the tasks except `myoChallengeDieReorient` and `myoChallengeBaoding`.
* Model is trained with enough epochs to understand and perform the task but could have performed better with more episodes.
