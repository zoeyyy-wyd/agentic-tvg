# Rollout analysis: RFT vs GRPO

## Aggregate

| | RFT | GRPO |
|---|---:|---:|
| n | 114 | 114 |
| format_score | 0.5000 | 0.5000 |
| answered | 1.0000 | 1.0000 |
| acc | 0.5044 | 0.5044 |
| evidence_iou | 0.2206 | 0.2178 |
| num_tool_calls | 1.0000 | 1.0000 |
| score | 1.1147 | 1.1133 |

RFT: tool calls 0/1/2/3 = 0/114/0/0 rows, mean response 2126 chars

GRPO: tool calls 0/1/2/3 = 0/114/0/0 rows, mean response 2044 chars

## Error taxonomy (acc x evidence window)

**RFT** (grounded = evidence_iou >= 0.3):

| judge | grounded | not grounded | reading |
|---|---:|---:|---|
| correct | 13 | 21 | right window -> right answer / lucky prior |
| partial | 19 | 28 |  |
| wrong | 5 | 28 | perception miss <-> localization miss |

**GRPO** (grounded = evidence_iou >= 0.3):

| judge | grounded | not grounded | reading |
|---|---:|---:|---|
| correct | 13 | 18 | right window -> right answer / lucky prior |
| partial | 20 | 33 |  |
| wrong | 5 | 25 | perception miss <-> localization miss |

## Flipped rows (RFT vs GRPO, paired on prompt: 114)

### RFT better: 19 rows

- **Q:** What festive decoration do a person's hands start to attach across the upper part of the window frame?
  - GT: A golden, shimmering tinsel garland.
  - RFT: acc 1.0, iou 0.19, crop [10,100] -> A long, shimmering, gold-colored tinsel garland.
  - GRPO: acc 0.5, iou 0.41, crop [15,50] -> A golden tinsel garland.
- **Q:** What happens to the leaf pile after the girl in the pink shirt emerges from it?
  - GT: A person in a dark hooded jacket then jumps into the leaf pile.
  - RFT: acc 0.5, iou 0.50, crop [28,48] -> The leaf pile is disturbed and slightly reduced in size.
  - GRPO: acc 0.0, iou 0.31, crop [28,60] -> The leaf pile is actively raked and compacted by another person.
- **Q:** What is the immediate consequence after the wakeboarder maneuvers over the first white ramp visible in the water?
  - GT: The wakeboarder falls into the water and gets submerged.
  - RFT: acc 0.5, iou 0.00, crop [40,50] -> The wakeboarder falls into the water.
  - GRPO: acc 0.0, iou 0.10, crop [20,100] -> The wakeboarder lands back on the water and continues riding.
- **Q:** After the green grater is placed on the counter, what yellow food item is repeatedly sliced with a knife?
  - GT: A block of yellow cheese.
  - RFT: acc 1.0, iou 0.66, crop [260,301] -> A block of yellow cheese.
  - GRPO: acc 0.0, iou 0.00, crop [160,210] -> A carrot.
- **Q:** What object does the child use to stand taller at the kitchen sink?
  - GT: The child uses a white step stool.
  - RFT: acc 1.0, iou 0.68, crop [45,60] -> A small white footstool.
  - GRPO: acc 0.5, iou 0.15, crop [0,151] -> A step stool.
- **Q:** After sectioning off the top layer of hair with a clip, what two actions does the person perform before blow-drying the bottom layer?
  - GT: They spray heat protector on their hair and then grab a hairdryer and a round patterned brush.
  - RFT: acc 1.0, iou 0.45, crop [30,50] -> They apply a heat protector and grab a hairdryer and a round-barreled brush.
  - GRPO: acc 0.5, iou 0.45, crop [30,50] -> They apply heat protectant spray and use a round brush to smooth the hair.
- **Q:** What does the person do with the red strap-like object retrieved from the small white table?
  - GT: The person attaches the red strap-like object to the small white dog.
  - RFT: acc 1.0, iou 0.76, crop [120,160] -> They use it to leash the dog.
  - GRPO: acc 0.5, iou 0.56, crop [130,160] -> They use it to tie the dog's leash.
- **Q:** After the man in the orange and black scarf looks down towards his side, what object does he bring to his mouth?
  - GT: He brings a black and green can to his mouth.
  - RFT: acc 0.5, iou 0.00, crop [30,40] -> A metallic can.
  - GRPO: acc 0.0, iou 0.00, crop [10,20] -> A piece of food.
- **Q:** After the right knitting needle pulls a new orange yarn loop through an existing stitch, what happens to the original stitch on the left needle?
  - GT: It is slid off the left needle.
  - RFT: acc 1.0, iou 0.00, crop [40,120] -> It is slipped off the left needle and onto the right needle.
  - GRPO: acc 0.5, iou 0.00, crop [40,100] -> The original stitch is removed and becomes a new, smaller loop on the right needle.
- **Q:** What trick does the person in the blue shirt perform on the slackline?
  - GT: They perform a split.
  - RFT: acc 0.5, iou 0.50, crop [60,120] -> He performs a handstand on the slackline and a full split while maintaining balance.
  - GRPO: acc 0.0, iou 0.30, crop [0,100] -> He performs a backflip on the slackline.
- **Q:** What emerges from the water after only a fin is visible?
  - GT: A large, disc-shaped fish.
  - RFT: acc 1.0, iou 0.08, crop [0,25] -> A large, unusual fish, resembling a sunfish, emerges from the water.
  - GRPO: acc 0.5, iou 0.13, crop [0,15] -> A large, unusual fish.
- **Q:** What colors are the top and pole used by the athlete who performs the first successful pole vault?
  - GT: The athlete wears a white top and uses a white or light grey pole.
  - RFT: acc 0.5, iou 0.54, crop [0,20] -> The pole is yellow and the top is white.
  - GRPO: acc 0.0, iou 0.25, crop [0,12] -> The pole is yellow, and the top is yellow.
- **Q:** What action does the player in the red shirt perform immediately after the player in the blue shirt serves the ball?
  - GT: The player in the red shirt returns the serve by hitting the ball with their paddle.
  - RFT: acc 1.0, iou 0.00, crop [0,10] -> The player in the red shirt returns the ball with a stroke.
  - GRPO: acc 0.5, iou 0.00, crop [0,10] -> The player in the red shirt hits the ball with a forehand shot.
- **Q:** What two eating utensils are primarily used to consume the food from the large black bowl?
  - GT: Chopsticks and a red spoon are used.
  - RFT: acc 1.0, iou 0.07, crop [120,190] -> Chopsticks and a red plastic spoon.
  - GRPO: acc 0.5, iou 0.25, crop [170,190] -> Chopsticks and a spoon.
- **Q:** What kind of aerial maneuver does the person perform immediately after jumping off the trampoline?
  - GT: The person performs a front flip.
  - RFT: acc 0.5, iou 0.00, crop [0,10] -> A complex flip in mid-air.
  - GRPO: acc 0.0, iou 0.00, crop [5,15] -> A backflip.
- **Q:** What activity is the Boston Terrier performing after the small, light brown dog runs through the grass?
  - GT: It is lying in the grass, panting.
  - RFT: acc 1.0, iou 0.50, crop [160,170] -> It is lying down on the grass, panting with its tongue out.
  - GRPO: acc 0.0, iou 0.00, crop [120,130] -> It is playing with a frisbee.
- **Q:** What action does the scuba diver perform after reaching the submerged vehicle?
  - GT: The diver enters the vehicle and sits behind the steering wheel.
  - RFT: acc 0.5, iou 0.30, crop [50,100] -> They examine the interior of the vehicle, including the steering wheel and dashboard.
  - GRPO: acc 0.0, iou 0.22, crop [55,100] -> They move away from the vehicle and continue exploring the underwater environment.
- **Q:** What happens to the red, green, and blue kite after the man in the striped shirt first attempts to launch it?
  - GT: It initially falls to the ground, but he then successfully relaunches it into the sky.
  - RFT: acc 0.5, iou 0.23, crop [0,100] -> The kite successfully takes flight and soars high into the sky.
  - GRPO: acc 0.0, iou 0.23, crop [0,100] -> It successfully takes flight and is flown by the man in the striped shirt.
- **Q:** What is done with the red long-sleeved top and blue jeans after they are laid out together on the bed?
  - GT: They are photographed with a cell phone.
  - RFT: acc 1.0, iou 0.00, crop [120,170] -> The red long-sleeved top and blue jeans are laid out together on the bed and then photographed.
  - GRPO: acc 0.5, iou 0.00, crop [100,200] -> The outfit is photographed and then folded and placed back into the red shopping bag.

### RFT worse: 16 rows

- **Q:** What happens to the white ATV after the boy completes a jump and continues riding it on the dirt track?
  - GT: The white ATV tips over onto its left side.
  - RFT: acc 0.0, iou 0.02, crop [40,100] -> It continues to ride independently on the dirt track, navigating the terrain and passing by various obstacles and parked vehicles.
  - GRPO: acc 0.5, iou 0.07, crop [40,55] -> The boy rides the white ATV, completes a jump, and continues riding it on the dirt track.
- **Q:** What does the wakeboarder do after ascending the white ramp and becoming airborne?
  - GT: The wakeboarder descends, lands back on the water, and continues to ride away.
  - RFT: acc 0.0, iou 0.00, crop [40,55] -> He performs a backflip and other aerial tricks.
  - GRPO: acc 0.5, iou 0.00, crop [30,100] -> He performs a series of dynamic maneuvers, including spins and turns, while maintaining control and balance on the board.
- **Q:** What is done to the round food item in the left pan using two utensils?
  - GT: It is flipped over.
  - RFT: acc 0.0, iou 0.00, crop [100,180] -> The person presses and shapes the round food item in the left pan using two utensils.
  - GRPO: acc 0.5, iou 0.07, crop [20,180] -> The person flips and presses down the round food item.
- **Q:** How does the camera's framing of the boat and ocean change after the initial low-angle view?
  - GT: The camera shifts to a wider perspective, showing more of the boat's bow, hull, and the entire sail as it moves through the waves.
  - RFT: acc 0.5, iou 0.00, crop [0,100] -> The camera moves from a close, low-angle view of the boat's mast and sails to a wider, eye-level perspective that reveals more of the boat's deck and the surrou
  - GRPO: acc 1.0, iou 0.00, crop [0,120] -> The camera moves from a close, low-angle view to a wider, elevated perspective, showing more of the ocean and the boat's structure.
- **Q:** What happens to the skier wearing a blue jacket and light blue pants as they move down the snowy trail?
  - GT: The skier loses balance and falls onto the snowy trail.
  - RFT: acc 0.0, iou 0.25, crop [12,100] -> The skier in the blue jacket and light blue pants is seen moving down the snowy trail, using ski poles for propulsion and balance, and interacting with other sk
  - GRPO: acc 0.5, iou 0.22, crop [0,100] -> The skier in the blue jacket and light blue pants is seen moving down the snowy trail, navigating through the trees and occasionally interacting with others.
- **Q:** What happens after the raft goes over the biggest drop in the river?
  - GT: The camera is submerged underwater, showing blurry orange and brown, before resurfacing.
  - RFT: acc 0.0, iou 0.67, crop [90,110] -> The raft continues to move downstream through calmer sections of the river.
  - GRPO: acc 0.5, iou 0.83, crop [90,120] -> The raft continues to navigate the river downstream, paddling through turbulent waters.
- **Q:** What food is the man primarily eating with his fork while seated at the table?
  - GT: He is eating scrambled eggs.
  - RFT: acc 0.5, iou 0.33, crop [150,162] -> Scrambled eggs and meat.
  - GRPO: acc 1.0, iou 0.32, crop [150,163] -> Scrambled eggs
- **Q:** What is the first item a person removes from the kitchen sink?
  - GT: A white mug.
  - RFT: acc 0.0, iou 0.21, crop [0,10] -> A large, dark-colored pot.
  - GRPO: acc 1.0, iou 0.21, crop [0,10] -> A white mug.
- **Q:** What red and yellow object is being cleaned with water after the person exits the garage?
  - GT: A red and yellow toy pedal car is being cleaned.
  - RFT: acc 0.5, iou 0.28, crop [200,300] -> A red and yellow toy car.
  - GRPO: acc 1.0, iou 0.28, crop [190,290] -> A red and yellow ride-on toy car.
- **Q:** What happens immediately after the practitioner wearing the red chest protector performs a high kick towards their opponent?
  - GT: The opponent, wearing a blue chest protector, falls to the ground.
  - RFT: acc 0.0, iou 0.00, crop [40,50] -> The opponent attempts to block or counter the kick, and the practitioner in the red chest protector maintains their offensive stance.
  - GRPO: acc 1.0, iou 0.00, crop [40,50] -> The opponent falls to the ground.
- **Q:** How many musicians are playing the series of dark, barrel-shaped drums arranged in a row?
  - GT: Three musicians are playing the drums.
  - RFT: acc 0.0, iou 0.24, crop [0,100] -> Four musicians
  - GRPO: acc 1.0, iou 0.60, crop [0,40] -> Three
- **Q:** What item does the girl sitting on the edge of the pool put on her face?
  - GT: She puts on her swimming goggles.
  - RFT: acc 0.0, iou 0.00, crop [100,150] -> A blue swim cap.
  - GRPO: acc 1.0, iou 0.00, crop [180,190] -> Swimming goggles
- **Q:** What color swimsuit is the person wearing while riding the large wave on the surfboard?
  - GT: The person is wearing a red swimsuit.
  - RFT: acc 0.0, iou 0.11, crop [30,100] -> Orange
  - GRPO: acc 1.0, iou 0.53, crop [40,55] -> Red
- **Q:** What change in visual content is observable across the entire clip?
  - GT: No change in visual content is observable, as all frames uniformly display a black screen.
  - RFT: acc 0.0, iou 0.19, crop [0,199] -> The video transitions from vibrant, active performance to a static, black void.
  - GRPO: acc 0.5, iou 0.00, crop [100,105] -> The video transitions to a black screen.
- **Q:** What happens to the arm of the man in the black shirt after it is nearly pinned down by his opponent during their arm wrestling match?
  - GT: He manages to push his opponent's arm down, winning the match.
  - RFT: acc 0.0, iou 0.38, crop [40,50] -> The arm snaps and breaks.
  - GRPO: acc 0.5, iou 0.38, crop [40,50] -> The arm is released and the man stands up.
- **Q:** What is placed on the mannequin that was previously wearing a black hooded garment, after the hooded person inspects it?
  - GT: A red and black plaid shirt is placed on the mannequin.
  - RFT: acc 0.0, iou 0.10, crop [250,300] -> A dark blue scarf.
  - GRPO: acc 0.5, iou 0.00, crop [130,170] -> A red and black patterned garment.

