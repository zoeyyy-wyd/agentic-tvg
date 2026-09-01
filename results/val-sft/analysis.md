# Rollout analysis: SFT vs GRPO

SFT rows = `results/grpo-vanilla/val_rollouts_grpo267/0.jsonl` (the GRPO
run's val_before_train — the SFT starting checkpoint, identical
prompts/decoding); GRPO rows = `267.jsonl` (final checkpoint). Same v1
judge verdicts as recorded at run time — the companion to
`results/val-rft/analysis.md` (RFT vs GRPO). Generated 2026-09-01.

## Aggregate

| | SFT | GRPO |
|---|---:|---:|
| n | 114 | 114 |
| format_score | 0.4868 | 0.5000 |
| answered | 0.9737 | 1.0000 |
| acc | 0.4561 | 0.5044 |
| evidence_iou | 0.1554 | 0.2178 |
| num_tool_calls | 0.9825 | 1.0000 |
| score | 1.0207 | 1.1133 |

SFT: tool calls 0/1/2/3 = 2/112/0/0 rows, mean response 4332 chars

GRPO: tool calls 0/1/2/3 = 0/114/0/0 rows, mean response 2044 chars

## Error taxonomy (acc x evidence window)

**SFT** (grounded = evidence_iou >= 0.3):

| judge | grounded | not grounded | reading |
|---|---:|---:|---|
| correct | 6 | 19 | right window -> right answer / lucky prior |
| partial | 14 | 40 |  |
| wrong | 5 | 30 | perception miss <-> localization miss |

**GRPO** (grounded = evidence_iou >= 0.3):

| judge | grounded | not grounded | reading |
|---|---:|---:|---|
| correct | 13 | 18 | right window -> right answer / lucky prior |
| partial | 20 | 33 |  |
| wrong | 5 | 25 | perception miss <-> localization miss |

## Flipped rows (SFT vs GRPO, paired on prompt: 114)

### SFT better: 14 rows

- **Q:** After the camels with riders approach the structured tents, what do the animals and their riders do next?
  - GT: The camels kneel down to the ground, and their riders dismount.
  - SFT: acc 0.5, iou 0.00, crop [25,100] -> They dismount and remount the camels, then continue walking across the desert.
  - GRPO: acc 0.0, iou 0.37, crop [120,180] -> They walk across the sandy desert landscape.
- **Q:** What is the initial action performed with the right knitting needle after the blue ribbed swatch is displayed without movement?
  - GT: The right knitting needle is inserted into the first stitch on the left needle.
  - SFT: acc 1.0, iou 0.00, crop [0,10] -> The right needle is inserted into the first stitch on the left needle.
  - GRPO: acc 0.5, iou 0.00, crop [0,10] -> The right knitting needle is inserted into a stitch and pulls the yarn through to create a loop.
- **Q:** What happens to the leaf pile after the girl in the pink shirt emerges from it?
  - GT: A person in a dark hooded jacket then jumps into the leaf pile.
  - SFT: acc 0.5, iou 0.67, crop [30,40] -> The leaf pile is slightly reduced in size and more disorganized.
  - GRPO: acc 0.0, iou 0.31, crop [28,60] -> The leaf pile is actively raked and compacted by another person.
- **Q:** After the green grater is placed on the counter, what yellow food item is repeatedly sliced with a knife?
  - GT: A block of yellow cheese.
  - SFT: acc 0.5, iou 0.14, crop [260,270] -> A block of cheese.
  - GRPO: acc 0.0, iou 0.00, crop [160,210] -> A carrot.
- **Q:** What does the person do with the red strap-like object retrieved from the small white table?
  - GT: The person attaches the red strap-like object to the small white dog.
  - SFT: acc 1.0, iou 0.62, crop [120,150] -> They put it on the dog's neck.
  - GRPO: acc 0.5, iou 0.56, crop [130,160] -> They use it to tie the dog's leash.
- **Q:** What does the woman in the light blue shirt hand to the sumo wrestler she was riding on?
  - GT: She hands him green sheets of paper.
  - SFT: acc 0.5, iou 0.03, crop [0,100] -> A piece of paper or a card.
  - GRPO: acc 0.0, iou 0.00, crop [20,30] -> A microphone.
- **Q:** After the right knitting needle pulls a new orange yarn loop through an existing stitch, what happens to the original stitch on the left needle?
  - GT: It is slid off the left needle.
  - SFT: acc 1.0, iou 0.00, crop [40,50] -> The original stitch is removed from the left needle.
  - GRPO: acc 0.5, iou 0.00, crop [40,100] -> The original stitch is removed and becomes a new, smaller loop on the right needle.
- **Q:** What trick does the person in the blue shirt perform on the slackline?
  - GT: They perform a split.
  - SFT: acc 1.0, iou 0.60, crop [50,100] -> They perform a split while balancing on the slackline.
  - GRPO: acc 0.0, iou 0.30, crop [0,100] -> He performs a backflip on the slackline.
- **Q:** What colors are the top and pole used by the athlete who performs the first successful pole vault?
  - GT: The athlete wears a white top and uses a white or light grey pole.
  - SFT: acc 0.5, iou 0.18, crop [0,10] -> White top and blue pole.
  - GRPO: acc 0.0, iou 0.25, crop [0,12] -> The pole is yellow, and the top is yellow.
- **Q:** What kind of aerial maneuver does the person perform immediately after jumping off the trampoline?
  - GT: The person performs a front flip.
  - SFT: acc 0.5, iou 0.00, crop [0,10] -> A series of flips and twists in the air.
  - GRPO: acc 0.0, iou 0.00, crop [5,15] -> A backflip.
- **Q:** What activity is the Boston Terrier performing after the small, light brown dog runs through the grass?
  - GT: It is lying in the grass, panting.
  - SFT: acc 1.0, iou 0.50, crop [160,170] -> It is lying on the grass, panting with its tongue out.
  - GRPO: acc 0.0, iou 0.00, crop [120,130] -> It is playing with a frisbee.
- **Q:** What action does the scuba diver perform after reaching the submerged vehicle?
  - GT: The diver enters the vehicle and sits behind the steering wheel.
  - SFT: acc 0.5, iou 0.56, crop [50,70] -> They examine the vehicle's interior, including the steering wheel and dashboard.
  - GRPO: acc 0.0, iou 0.22, crop [55,100] -> They move away from the vehicle and continue exploring the underwater environment.
- **Q:** After the diver moves further inside from the large rectangular opening, what animal swims into view through it?
  - GT: A yellowish-orange fish swims into view.
  - SFT: acc 0.5, iou 0.00, crop [60,75] -> A fish.
  - GRPO: acc 0.0, iou 0.00, crop [50,100] -> A large, dark-colored shark.
- **Q:** After the person brings the cup towards their mouth and tilts it for a drink, what is visible inside the cup when it is lowered again?
  - GT: Only a small amount of dark residue or grounds is visible at the bottom of the cup.
  - SFT: acc 0.5, iou 0.00, crop [120,130] -> A dark brown liquid.
  - GRPO: acc 0.0, iou 0.00, crop [15,30] -> It contains a light brown liquid, likely coffee or tea.

### SFT worse: 23 rows

- **Q:** What happens after the man in the white shirt and brown pants falls backward off the slackline for the third time?
  - GT: Another man in a yellow shirt walks over and helps him up from the ground.
  - SFT: acc 0.0, iou 0.36, crop [120,130] -> He gets up and continues walking on the slackline.
  - GRPO: acc 0.5, iou 0.45, crop [120,150] -> He gets up and walks away from the slackline, then interacts with another person.
- **Q:** What two distinct tools does the stylist consistently use in conjunction to blow dry and style the client's hair?
  - GT: A red hairdryer and a black round brush.
  - SFT: acc 0.5, iou 0.10, crop [0,100] -> A red hairdryer and a round brush.
  - GRPO: acc 1.0, iou 0.06, crop [0,161] -> A red hairdryer and a round hairbrush.
- **Q:** What happens to the young skateboarder in the white shirt and helmet after he attempts to ride off the raised wooden platform?
  - GT: He loses balance, falls to the ground, and then gets back up.
  - SFT: acc 0.0, iou 0.00, crop [0,12] -> He successfully rides off the platform and continues skating down the ramp.
  - GRPO: acc 0.5, iou 0.80, crop [10,20] -> He falls off the platform and lands on his hands and knees.
- **Q:** What happens to the white ATV after the boy completes a jump and continues riding it on the dirt track?
  - GT: The white ATV tips over onto its left side.
  - SFT: acc 0.0, iou 0.00, crop (no crop) -> (no answer tag)
  - GRPO: acc 0.5, iou 0.07, crop [40,55] -> The boy rides the white ATV, completes a jump, and continues riding it on the dirt track.
- **Q:** What action does the child perform while wearing a patterned skirt and a pink top?
  - GT: The child walks up the stairs.
  - SFT: acc 0.5, iou 0.00, crop [100,150] -> The child runs up the staircase.
  - GRPO: acc 1.0, iou 0.00, crop [100,160] -> The child climbs the staircase while holding onto the railing.
- **Q:** What is done to the round food item in the left pan using two utensils?
  - GT: It is flipped over.
  - SFT: acc 0.0, iou 0.11, crop [10,120] -> The person uses two utensils to press and flatten the round food item in the left pan.
  - GRPO: acc 0.5, iou 0.07, crop [20,180] -> The person flips and presses down the round food item.
- **Q:** How does the camera's framing of the boat and ocean change after the initial low-angle view?
  - GT: The camera shifts to a wider perspective, showing more of the boat's bow, hull, and the entire sail as it moves through the waves.
  - SFT: acc 0.5, iou 0.00, crop [10,60] -> The camera shifts to a more stable, elevated perspective, offering a broader view of the open sea and the boat's structure.
  - GRPO: acc 1.0, iou 0.00, crop [0,120] -> The camera moves from a close, low-angle view to a wider, elevated perspective, showing more of the ocean and the boat's structure.
- **Q:** What two types of hair are being blow-dried concurrently after the woman begins styling the blonde hair?
  - GT: Blonde hair and dark brown hair.
  - SFT: acc 0.5, iou 0.00, crop [30,50] -> Blonde and black hair
  - GRPO: acc 1.0, iou 0.24, crop [30,60] -> Blonde and dark hair
- **Q:** After completing the initial pencil outlines on the canvas, what action does the artist perform to start applying color?
  - GT: The artist picks up a paintbrush and starts applying a dark green base coat to the sketched tree areas.
  - SFT: acc 0.5, iou 0.45, crop [50,60] -> He begins painting the trees with a brush.
  - GRPO: acc 1.0, iou 0.37, crop [10,70] -> The artist uses a brush to apply dark green color to the trees.
- **Q:** What happens to the skier wearing a blue jacket and light blue pants as they move down the snowy trail?
  - GT: The skier loses balance and falls onto the snowy trail.
  - SFT: acc 0.0, iou 0.20, crop [10,120] -> The skier in the blue jacket and light blue pants is seen actively skiing down the snowy trail, making turns and occasionally interacting with others.
  - GRPO: acc 0.5, iou 0.22, crop [0,100] -> The skier in the blue jacket and light blue pants is seen moving down the snowy trail, navigating through the trees and occasionally interacting with others.
- **Q:** What food is the man primarily eating with his fork while seated at the table?
  - GT: He is eating scrambled eggs.
  - SFT: acc 0.0, iou 0.00, crop [140,150] -> He is eating a banana.
  - GRPO: acc 1.0, iou 0.32, crop [150,163] -> Scrambled eggs
- **Q:** What red and yellow object is being cleaned with water after the person exits the garage?
  - GT: A red and yellow toy pedal car is being cleaned.
  - SFT: acc 0.5, iou 0.00, crop [210,220] -> A red and yellow toy car.
  - GRPO: acc 1.0, iou 0.28, crop [190,290] -> A red and yellow ride-on toy car.
- **Q:** What happens immediately after the practitioner wearing the red chest protector performs a high kick towards their opponent?
  - GT: The opponent, wearing a blue chest protector, falls to the ground.
  - SFT: acc 0.0, iou 0.00, crop [40,50] -> The practitioner in the red chest protector advances towards their opponent.
  - GRPO: acc 1.0, iou 0.00, crop [40,50] -> The opponent falls to the ground.
- **Q:** What action does the person on rollerblades perform when approaching the stairs?
  - GT: The person jumps up the stairs while on rollerblades.
  - SFT: acc 0.0, iou 0.00, crop [50,60] -> They descend the stairs smoothly.
  - GRPO: acc 0.5, iou 0.91, crop [70,90] -> They rollerblade up the stairs.
- **Q:** How many musicians are playing the series of dark, barrel-shaped drums arranged in a row?
  - GT: Three musicians are playing the drums.
  - SFT: acc 0.0, iou 0.48, crop [0,50] -> Four
  - GRPO: acc 1.0, iou 0.60, crop [0,40] -> Three
- **Q:** What action do the players perform immediately after several colored dodgeballs are lined up on the court?
  - GT: They run from opposing sides to gather the dodgeballs in the center.
  - SFT: acc 0.0, iou 0.23, crop [20,100] -> They throw and dodge the balls.
  - GRPO: acc 0.5, iou 0.36, crop [0,30] -> They begin to run and throw the balls at each other.
- **Q:** What emerges from the water after only a fin is visible?
  - GT: A large, disc-shaped fish.
  - SFT: acc 0.0, iou 0.00, crop (no crop) -> (no answer tag)
  - GRPO: acc 0.5, iou 0.13, crop [0,15] -> A large, unusual fish.
- **Q:** After lifting the kite control bar and lines high above their head, what action does the person perform to get the kite into the air?
  - GT: They run forward across the grassy field until the kite becomes airborne.
  - SFT: acc 0.0, iou 0.42, crop [10,20] -> They pull down on the lines.
  - GRPO: acc 0.5, iou 0.42, crop [10,20] -> They run forward while holding the bar high.
- **Q:** What color swimsuit is the person wearing while riding the large wave on the surfboard?
  - GT: The person is wearing a red swimsuit.
  - SFT: acc 0.0, iou 0.80, crop [40,50] -> White
  - GRPO: acc 1.0, iou 0.53, crop [40,55] -> Red
- **Q:** What change in visual content is observable across the entire clip?
  - GT: No change in visual content is observable, as all frames uniformly display a black screen.
  - SFT: acc 0.0, iou 0.00, crop [0,120] -> The video transitions from a vibrant, active performance to a static, black void.
  - GRPO: acc 0.5, iou 0.00, crop [100,105] -> The video transitions to a black screen.
- **Q:** What does the person do on the window immediately after finishing cleaning an indoor section and stepping out onto the patio?
  - GT: He cleans the lower section of the window from the exterior.
  - SFT: acc 0.5, iou 0.49, crop [100,120] -> They begin cleaning the window from the patio.
  - GRPO: acc 1.0, iou 0.36, crop [100,130] -> They begin cleaning the window from the outdoor perspective.
- **Q:** What happens to the runner wearing bib 258 as she is running on the muddy ground?
  - GT: She stumbles and falls to the ground, then multiple people come to help her stand up.
  - SFT: acc 0.0, iou 0.00, crop [40,50] -> She continues running on the muddy road.
  - GRPO: acc 0.5, iou 0.00, crop [60,100] -> She loses her footing and falls, sliding on the mud.
- **Q:** What is placed on the mannequin that was previously wearing a black hooded garment, after the hooded person inspects it?
  - GT: A red and black plaid shirt is placed on the mannequin.
  - SFT: acc 0.0, iou 0.00, crop [120,130] -> A red and white striped hat.
  - GRPO: acc 0.5, iou 0.00, crop [130,170] -> A red and black patterned garment.

