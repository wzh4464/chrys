# Buddy Visual Design — Complete 28 Species Reference

## Design Principles

1. **Silhouette recognition** — each species must be identifiable from its outline alone, before eyes/mouth are rendered
2. **Species-authentic faces** — mouths reflect real biology or fun subversions (shark teeth, snake tongue, robot speaker)
3. **Distinct idle animation** — no two species share the same animation pattern
4. **No identical frames** — every species has at least 2 visually distinct frames; zero-animation species eliminated

---

## Avian (4 species)

### Duck
Silhouette: round head, flat bill profile, compact body, pointed tail

```
Frame 0 (rest):     Frame 1 (bob):       Frame 2 (quack):
                         ~
     __                   __                   __
  <(·· )____          <(·· )____          <(··°)____
   (  __ )              (  __ )              (  __ )
    `--´                 `--´                 `--´

Mouth: __ (flat bill, horizontal)    Eye: ·· → DOT preferred
Unique: Bill opens (°) on frame 2 for quack; tail tilde on frame 1 for bob
```

### Goose
Silhouette: long curved neck, heavy body, wide tail

```
Frame 0 (rest):     Frame 1 (honk-pre):  Frame 2 (honk):

      __                   __                   __
   __(··)___           __(··)___           __(··°)___
  (  >.°  )           (  >.°  )~          (  >.°° )
   `----´               `----´               `----´

Mouth: ° (open round beak)             Eye: ·· → DOT preferred
Unique: Longest neck of all birds; beak fully opens (°°) on frame 2 for honk
Distinct from Duck: longer neck, different body/tail profile, no flat bill
```

### Owl
Silhouette: broad facial disc, ear tufts, compact body on perch

```
Frame 0 (rest):     Frame 1 (tilt-R):    Frame 2 (tilt-L):

   /^---^\             /^---^\             /^---^\
  │ ··  │            │ ·· ~│            ~│ ··  │
  │  v  │            │  v  │            │  v  │
   \---/               \---/               \---/
    ▔▔                  ▔▔                  ▔▔

Mouth: v (tiny beak within facial disc)  Eye: ·· → BULLSEYE preferred (big owl eyes)
Unique: Ear tufts (^) on top; head tilts side-to-side (frames 1-2 shift ~ position)
Facial disc is the defining silhouette feature
```

### Penguin
Silhouette: upright posture, flipper wings, tuxedo contrast

```
Frame 0 (rest):     Frame 1 (waddle-R):  Frame 2 (waddle-L):

    ___                 ___                 ___
   │··│               │··│~              ~│··│
   │ >│               │ >│               │ >│
   │__│               │__│               │__│
   ▔▔▔▔               ▔▔▔▔               ▔▔▔▔

Mouth: > (pointy beak, forward-facing)   Eye: ·· → DOT preferred
Unique: Tuxedo body (solid block with contrast belly); waddles side-to-side
```

---

## Aquatic (6 species)

### Shark
Silhouette: dorsal fin, streamlined torpedo body, crescent tail

```
Frame 0 (swim1):    Frame 1 (swim2):     Frame 2 (jaw-open):

      ^                   ^                    ^
  ____|___            ____|___             ____|___
 <··  ﹏  )         <··  ﹏  )~        <··  ﹏﹏ )
  ‾‾‾‾‾‾‾‾          ‾‾‾‾‾‾‾‾           ‾‾‾‾‾‾‾‾

Mouth: ﹏ (jagged tooth line)            Eye: ·· → DOT preferred (dead-fish-eye)
Unique: Dorsal fin (^) is the defining silhouette; frame 2 shows extra teeth (﹏﹏)
Always in motion — 3-frame swim loop with jaw opening
```

### Octopus
Silhouette: bulbous head, eight tentacles splayed below

```
Frame 0 (rest):     Frame 1 (wave-R):    Frame 2 (wave-L):

   .---.               .---.               .---.
  ( ·· )             ( ·· )             ( ·· )
 (  ~ )~~~           (  ~ )~~ ~          (  ~ )~ ~~
  `-----´             `-----´             `-----´

Mouth: ~ (siphon funnel among tentacles)  Eye: ·· → DOT preferred
Unique: Tentacle arrangement (~~~) shifts direction each frame;
mouth position is central siphon, not a facial mouth
```

### Crab
Silhouette: wide horizontal body, two large claws, eye stalks

```
Frame 0 (rest):     Frame 1 (claw-L):    Frame 2 (claw-R):

 (V) (··) (V)       (v) (··) (V)        (V) (··) (v)
  \__ ﹀ __/         \__ ﹀ __/          \__ ﹀ __/
     \__/               \__/                \__/~
    /    \             /    \              /    \

Mouth: ﹀ (small mouth under carapace)    Eye: ·· → DOT preferred
Unique: Claws (V) are the defining feature; frame 1 snaps left claw (v),
frame 2 snaps right claw; eye stalks position eyes ABOVE the body line
```

### Axolotl
Silhouette: frilly external gills on both sides of head, salamander body

```
Frame 0 (rest):     Frame 1 (gill-R):    Frame 2 (gill-L):

  ≈≈≈  ≈≈≈           ≈≈≈ ≈≈≈            ≈≈≈  ≈≈≈
   (··)               (··)                (··)
   ( v )~~~            ( v )~~~            ( v )~~ ~
    ‾‾‾                 ‾‾‾                 ‾‾‾

Mouth: v (perpetual smile)               Eye: ·· → DOT preferred
Unique: Gills (≈≈≈) are the defining silhouette — frilly branches on both sides
of head, NOT tentacles like octopus; gill pattern shifts between frames
```

### Turtle
Silhouette: domed shell, small protruding head and limbs

```
Frame 0 (rest):     Frame 1 (retreat):   Frame 2 (emerge):

   _____               _____               _____
  │··  │              │-·  │              │··  │
  │ v)_│              │ v)_│              │ v)_│~
   ‾‾‾‾                ‾‾‾‾                ‾‾‾‾

Mouth: v (tiny beak-mouth)               Eye: ·· → DOT preferred
Unique: Domed shell (_____) is the defining feature; frame 1 shows
half-closed eye (retreating into shell), frame 2 emerges fully with tail wiggle
```

### Frog
Silhouette: wide flat head, bulging eyes, compact body, long hind legs

```
Frame 0 (rest):     Frame 1 (croak):      Frame 2 (rest2):     Frame 3 (jump):

 (··)__(··)         (··)__(··)            (··)__(··)           (··)__(··)
 ( ﹀  ﹀ )         ( ﹀oo﹀ )           ( ﹀  ﹀ )           ( ﹀  ﹀ )
  (      )           (      )             (      )             (      )
   `----´             `----´               `----´               `----´
                                                                ··  (land splash)

Mouth: ﹀ (wide frog mouth spanning face width)  Eye: ·· → BULLSEYE preferred (bulging)
Unique: Frame 1 inflates throat sac (oo); frame 3 jumps with landing particles (··);
wide mouth is the defining facial feature
```

---

## Mammals (7 species)

### Cat
Silhouette: pointed ears, curved back, long tail

```
Frame 0 (rest):     Frame 1 (lick-paw):  Frame 2 (tail-flick):

   /\_/\               /\_/\               /\_/\
  ( ·· )             ( -· )↑             ( ·· )
  (  ω )             (  ω )             (  ω )~
   / /                / /                / /
  /_/                /_/                /_/

Mouth: ω (cat lips — classic feline mouth shape)  Eye: ·· → DOT/STAR preferred
Unique: Pointed ears (/\_/\) are the cat silhouette; frame 1 lifts paw with
one eye closed; frame 2 tail flick (~)
```

### Chonk
Silhouette: same as cat but much wider, no visible neck

```
Frame 0 (rest):     Frame 1 (sleepy):    Frame 2 (tail-flick):

   /\_/\               /\_/\               /\_/\
  ( ··· )             ( -·· )             ( ··· )
 (  ω    )           (  ω    )           (  ω    )~
  \_____/             \_____/             \_____/

Mouth: ω (same cat mouth)                Eye: ·· → DOT preferred
Unique: Extra-wide body (··· on eye line, \_____/ base); visibly wider than Cat;
frame 1 shows one eye half-closed (sleepy chonk)
```

### Fox
Silhouette: large triangular ears, pointy snout, enormous bushy tail

```
Frame 0 (rest):     Frame 1 (tail-R):    Frame 2 (tail-L):

   ^---^               ^---^               ^---^
  ( ·· )             ( ·· )             ( ·· )
  ( ﹀ )             ( ﹀ )~            ( ﹀ )
   / /  ~              / /                 / /~~
  /_/                /_/                /_/

Mouth: ﹀ (sly grin — wider and more cunning than cat ω)  Eye: ·· → STAR preferred
Unique: Large triangular ears (^---^); ENORMOUS tail (~ and ~~ on bottom rows);
frame 1 tail flicks right, frame 2 tail sweeps left dramatically
```

### Rabbit
Silhouette: extremely long ears, compact round body, cotton tail

```
Frame 0 (rest):     Frame 1 (ear-L):     Frame 2 (ear-R):

   │   │               \   │               │   /
   \··/                \··/                \··/
   ( Y )               ( Y )               ( Y )~
    / /                 / /                 / /
   /_/                 /_/                 /_/

Mouth: Y (bunny nose — Y-shaped cleft)   Eye: ·· → DOT preferred
Unique: Very long vertical ears (│   │) are the defining silhouette;
frame 1 left ear bends, frame 2 right ear bends; Y nose with possible wiggle
```

### Capybara
Silhouette: rectangular block body, small head, sitting pose

```
Frame 0 (rest):     Frame 1 (blink):     Frame 2 (tail-wiggle):

    ___                 ___                 ___
   │··│                │-·│                │··│
   │ _│━━━━            │ _│━━━━            │ _│━━━━~
    ‾‾                  ‾‾                  ‾‾

Mouth: _ (flat line — ultimate chill)    Eye: ·· → DEGREE preferred (half-closed)
Unique: Ultra-rectangular body silhouette (━━━━ for the horizontal block);
barely moves (capybaras are famously chill); frame 1 only closes ONE eye
```

### Panda
Silhouette: round body, round ears, sitting pose with visible eye patches

```
Frame 0 (rest):     Frame 1 (munch):     Frame 2 (blink):

   ○     ○             ○     ○             ○     ○
  (·)___(·)          (·)___(·)          (·)___(·)
  (  ··   )          (  ··   )          (  -·   )
  (  ﹀   )          (  ﹀   )~         (  ﹀   )
   ‾‾‾‾‾‾            ‾‾‾‾‾‾            ‾‾‾‾‾‾

Mouth: ﹀ (gentle smile)                 Eye: ·· → DOT preferred
Unique: Round ears (○) and eye patches (·) on either side are the defining panda
features; frame 1 munches bamboo; sitting pose with wide base
```

### Llama
Silhouette: extremely long vertical neck, small head, tall stance

```
Frame 0 (rest):     Frame 1 (neck-turn): Frame 2 (chew):

   (··)                (··)                (··)
   ( v )               ( v )               ( ~ )
   │  │____            │  │____ ~          │  │____
   │       │           │       │           │       │
   └─┘───└─┘           └─┘───└─┘           └─┘───└─┘

Mouth: v (slight smile)                  Eye: ·· → DOT preferred
Unique: EXTREMELY long neck (rows 0-1 are all head/neck) — tallest sprite;
frame 2 changes mouth to ~ for chewing motion; four long legs
```

---

## Fantasy/Mythical (3 species)

### Dragon
Silhouette: horns, wings, long fire-trailing tail

```
Frame 0 (rest):     Frame 1 (inhale):    Frame 2 (fire1):     Frame 3 (fire2):

    (\/)               (\/)~               (\/)                 (\/)
   ( ◉◉)             ( -◉)               ( ◉◉)               ( ◉◉)
   ( ﹀ )━━━          ( ﹀ )━━━           ( ﹀~~)              ( ﹀~~~)
    / /                / /                 / /                  / /
   /_/                /_/                 /_/                  /_/

Mouth: ﹀ with ~~ fire particles          Eye: ·· → BULLSEYE preferred (slit pupils)
Unique: Horns (\/) on head; fire-breathing is the signature — frame 1 inhales
(one eye closed, no flame), frame 2 small flame (~~), frame 3 big flame (~~~);
tail trail (━━━) extends behind
```

### Ghost
Silhouette: rounded top, wavy ethereal bottom, no legs

```
Frame 0 (solid):    Frame 1 (float-up):  Frame 2 (fade):

   .---.               .---.               .-≈-.
   │··│               │··│ ≈             │··│
   │ O│               │ O│               │ O│
   ≈≈≈≈               ≈≈≈≈               ≈≈≈≈

Mouth: O (round shocked/surprised — classic ghost expression)  Eye: ·· → DOT preferred
Unique: Wavy bottom (≈≈≈≈) is the ghost silhouette; frame 1 floats upward (≈ above);
frame 2 crown becomes wavy (.≈-≈.) for fade/transparency effect
```

### Alien
Silhouette: oversized head, tiny body, antenna

```
Frame 0 (rest):     Frame 1 (signal):    Frame 2 (stare):

    .---.               .---.               .---.
   / ··  \             / ··* \             / ◎◎  \
  │  ﹀   │           │  ﹀   │           │  ﹀   │
   \     /              \     /              \     /
    '---'               '---'               '---'

Mouth: ﹀ (tiny mysterious smile)         Eye: ·· → AT(@) preferred (alien eyes)
Unique: Oversized dome head (．---．) with tiny body below is THE alien silhouette;
frame 1 antenna blinks (*); frame 2 eyes enlarge (◎◎) for intense stare
```

---

## Insects & Arthropods (2 species)

### Bee
Silhouette: oval body with stripes, two wing pairs, antenna

```
Frame 0 (wings-up): Frame 1 (wings-dn):  Frame 2 (buzz):

   __  __              __  __              __  __
  (··)/  \            (··)\  /            (··)/  \~
 ( ~ )║║║║           ( ~ )║║║║           ( ~ )║║║║
  `--´\__/            `--´\__/            `--´\__/

Mouth: ~ (proboscis — drinking-straw mouth)  Eye: ·· → BULLSEYE preferred (compound)
Unique: Wings (__  __ ) and stripes (║║║║) are defining features;
frame 1 wings flip direction, frame 2 adds buzz vibration (~);
round fuzzy body with stinger (not shown — cute bee!)
```

### Snail
Silhouette: spiral shell on back, eye stalks, flat foot

```
Frame 0 (rest):     Frame 1 (stalk-up):  Frame 2 (stalk-dn):

     __                  __                  __
    /··\               /··\~              /··\
   │ v )               │ v )              │ v )~
    \__/                \__/               \__/

Mouth: v (tiny mouth under eye stalks)   Eye: ·· → DOT preferred
Unique: Shell (__ on top, \__/ at base) houses the body;
eyes are on stalks (/··\) at the TOP of the head; frame 1 stalks extend (~),
frame 2 stalks retract; extremely slow animation (mostly rest frames)
```

---

## Plants & Fungi (2 species)

### Cactus
Silhouette: vertical stem, arms, flower pot, spines

```
Frame 0 (rest):     Frame 1 (bloom):     Frame 2 (swell):

     │                   │*                   │
   ┌─┴─┐              ┌─┴─┐               ┌──┴──┐
   │·· │              │·· │               │ ·· │
   │   │              │   │               │    │
   └───┘              └───┘               └────┘

Mouth: NONE (plants don't have mouths!)  Eye: ·· → DOT preferred
Unique: VERTICAL STEM (│) is the defining silhouette; NO MOUTH — this is the
key differentiator from all other species; frame 1 blooms (*) on top;
frame 2 body swells wider; pot (└───┘) anchors the bottom
```

### Mushroom
Silhouette: wide domed cap, slender stem

```
Frame 0 (rest):     Frame 1 (spore1):    Frame 2 (bounce):

   .---.               .---.               .-≈-.
  │·· ·│              │·· ·│              │·· ·│
  │ o │               │ o │               │ o │
   └───┘               └───┘               └───┘
                          ·                  ··

Mouth: o (tiny round mouth under cap)    Eye: ·· → DOT preferred
Unique: Wide domed cap (．---．) over slender stem is the shroom silhouette;
frame 1 releases a single spore (·); frame 2 cap bounces (．-≈-．) with more spores (··)
```

---

## Other (4 species)

### Blob (Slime)
Silhouette: amorphous, constantly shifting shape

```
Frame 0 (rest):     Frame 1 (squish):    Frame 2 (stretch):

   ┌────┐              ┌──────┐             ┌────┐
   │·· │              │ ·· │              │·· │
   │ o │              │  o │              │  o │
   └────┘              └──────┘             └──┘

Mouth: o (small, sometimes shifts position)  Eye: ·· → DOT preferred
Unique: SHAPE-SHIFTING is the identity — frame 1 squishes wide (eyes and mouth
shift left), frame 2 stretches tall; mouth position is NOT fixed between frames
```

### Robot
Silhouette: rectangular chassis, antenna, rigid posture

```
Frame 0 (rest):     Frame 1 (signal):    Frame 2 (gear):

   ┌───┐               ┌───┐               ┌───┐
   │[=]│               │[=]│*              │[=]│
   │ □ │               │ □ │               │ □ │◷
   └───┘               └───┘               └───┘
    ‾‾                  ‾‾                  ‾‾

Mouth: □ (speaker grille)                Eye: [=] → AT(@) preferred (LED screens)
Unique: Boxed chassis (┌───┐/└───┘) is the rigid mechanical silhouette;
[=] LED eyes replaced with eye char at render time; frame 1 antenna flash (*);
frame 2 gear rotates (◷); only species with square brackets for eyes
```

### Snake
Silhouette: long sinuous body, no limbs, forked tongue

```
Frame 0 (coil1):    Frame 1 (coil2):     Frame 2 (coil3):     Frame 3 (tongue):

    (··)               (··)                 (··)                 (··)
     V \                V \~                 V \                  VV\
  \____/             \____/              \____/               \____/
  /                  /                   /                    /
  ‾‾                 ‾‾                  ‾‾                   ‾‾

Mouth: V (forked tongue)                 Eye: ·· → DOT preferred
Unique: LONG SINUOUS BODY is the defining feature — horizontal sprawl across
4-5 rows; frame 3 tongue becomes VV (double fork flick);
body segments shift each frame producing slither illusion
```

### Bat
Silhouette: wide wing membranes, small body, hanging upside-down

```
Frame 0 (hang):     Frame 1 (spread-L):  Frame 2 (spread-R):

   /--^--\             /--^--\             /----\
  / (··) \            / (··) \~           / (··) \
  \ (v ) /            \ (v ) /            \ (v ) /
   \    /              \    /              \    /
    \  /                \  /                \  /

Mouth: v (tiny mouth on small face)      Eye: ·· → BULLSEYE preferred (nocturnal)
Unique: WING MEMBRANES (/--^--\) are the defining silhouette — wide span;
frame 2 wings spread extra wide (/----\); ear points (^) on top
```

---

## Silhouette Recognition Summary

| Recognition | Species |
|---|---|
| Instant | Shark (fin), Snake (long), Llama (tall), Dragon (fire tail), Bee (wings+stripes), Crab (claws), Bat (wings), Frog (wide), Robot (box), Cactus (stem+pot), Penguin (tuxedo block), Blob (amorphous) |
| Distinct | Duck (flat bill), Owl (ear tufts), Rabbit (long ears), Chonk (extra wide), Fox (big tail), Turtle (domed shell), Mushroom (cap+stem), Octopus (tentacles), Axolotl (gills), Ghost (wavy bottom), Alien (dome head), Panda (eye patches) |
| Adequate | Cat (pointed ears), Goose (long neck), Capybara (block body), Snail (shell spiral) |

No species should fall below "Adequate" — every silhouette must be distinguishable.

---

## Implementation Notes

### Eye placeholder conventions
- `··` → `{eye_char}{eye_char}` for standard two-eye replacement
- `[=]` → `[{eye_char}]` for Robot's bracket-enclosed LED eyes
- Blink replaces eye_char with `-`

### Mouth conventions
- All mouths are hardcoded in sprite data (no runtime replacement needed)
- Each species mouth shown in design table above
- Mouth chars are Unicode: ω (U+03C9), ﹀ (U+FE40), ﹏ (U+FE4F), V (U+0056), etc.

### Hat slot
- Row 0 of every frame is the hat slot
- Hat is applied only when row 0 is blank (`.strip()` is empty)
- Species with non-blank row 0 (Duck's `__`, Goose's `__`, Owl's `/^---^\`, etc.) cannot wear hats on those frames

### Frame counts
- 3 frames: Most species (standard idle + 2 action variants)
- 4 frames: Species with complex animations (Dragon fire cycle, Snake slither + tongue, Frog croak + jump)
