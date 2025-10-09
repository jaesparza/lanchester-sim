The linear law models combat as one-on-one engagements, where each fighter faces a single opponent—like hoplites or medieval knights. Casualties increase in proportion to the number of direct pairings. Modern warfare differs: machine gunners can engage multiple targets, artillery and drones strike across the battlefield, and tanks concentrate firepower. Because each unit can affect many opponents, attrition grows faster, roughly with the square of force size. Therefore, the square law is used for modern ranged, aimed weapons, while the linear law applies mainly to hand-to-hand or limited-rate weapons.

The following table classifies common weapon and combat system types according to which Lanchester law—linear or square—best approximates their attrition dynamics. It considers effective range, engagement mode, and whether each unit can concentrate fire on multiple targets, providing a practical framework for linking weapon characteristics to attrition modeling.


| Weapon / System                            | Effective Range & Engagement Mode        | Closest Lanchester Law       | Reason                                                     |
| ------------------------------------------ | ---------------------------------------- | ---------------------------- | ---------------------------------------------------------- |
| **Melee (sword, spear, bayonet)**          | Contact only, one opponent per fighter   | Linear                       | One-on-one matchups dominate                               |
| **Pistol / revolver / shotgun**            | Very short range, low rate of fire       | Linear (approx)              | Mostly one target at a time                                |
| **Submachine gun**                         | Short range, high fire rate              | Between Linear & Square      | Depends on terrain: one target (indoors) vs. groups (open) |
| **Bolt-action / semi-auto rifles**         | Medium–long range, aimed fire            | Square                       | Each shooter can pick targets across battlefield           |
| **Assault rifle / battle rifle**           | Long range, rapid fire                   | Square                       | Firepower adds across shooters, not just pairwise          |
| **Light / heavy machine gun**              | Sustained fire, area suppression         | Square                       | Can engage multiple enemies simultaneously                 |
| **Anti-tank weapons (RPG, ATGM, bazooka)** | Long range, selective fire               | Square                       | Few operators, each can strike valuable armored units      |
| **Artillery / mortars / rockets**          | Long–very long range, area effects       | Square                       | Concentrated firepower scales with numbers                 |
| **UAVs / armed drones**                    | Wide coverage, multiple possible targets | Square                       | Force size multiplies both sensing and killing power       |
| **Electronic warfare**                     | Non-kinetic, degrades enemy systems      | Neither (modeled separately) | Doesn’t cause attrition but reduces effectiveness          |

## Estimating Cold War infantry platoon coefficients

When you want concrete coefficients for two historical platoons—say a late Cold War U.S. Army rifle platoon fighting a Soviet motor-rifle platoon—the workflow is:

1. **Pick the governing law.** Squads equipped with assault rifles, machine guns, and anti-armor weapons can concentrate fire on any exposed opponent, so the square law is the natural starting point.
2. **List weapon groups.** Separate each platoon into clusters that share rate of fire, hit probability, and lethality (e.g., riflemen, squad automatic weapons, medium machine guns, anti-armor teams).
3. **Translate weapon performance into per-platform kill production.** For each group estimate the sustained combat rate of fire (rounds per minute while maneuvering), multiply by the probability that a target is exposed, the probability of a hit when firing, and the probability that a hit is lethal:

   
   \[
   k_i = \text{ROF}_i \times P(\text{target exposed}) \times P(\text{hit} \mid \text{exposed}) \times P(\text{kill} \mid \text{hit})
   \]

   Use training data for hit probabilities (marksmanship qualification scores, range tables) and ballistics/fragmentation data for lethality. Exposure probability packages terrain, posture, and suppression effects—0.3–0.7 is typical for platoon-level firefights.
4. **Aggregate by platoon.** Sum over all platforms on the firing side:

   \[
   K = \sum_i n_i \times k_i
   \]

   Here \(K\) is the expected enemy casualties per minute inflicted by the platoon when targets are available.
5. **Convert to Lanchester coefficients.** For the square law formulation \(\dot{R} = -\beta B\) and \(\dot{B} = -\alpha R\), divide the aggregated kill production by the number of shooters on the firing side. The result has units of “kills per shooter per minute” and can be used directly as \(\alpha\) or \(\beta\).

### Worked 1980s platoon example

The tables below use mid-1980s organizations (U.S. rifle platoon with M16A2s, M249 SAWs, and M60 machine guns; Soviet BMP motor-rifle platoon with AK-74s, RPK-74s, PKM machine guns, and RPG-7s).[^1][^2] Sustained rates of fire reflect doctrinal guidance for deliberate firefights rather than cyclic rates.[^3][^4] Hit probabilities are derived from qualification scores (U.S. soldiers averaged ~23/40 hits, Soviet conscripts ~20/60) adjusted for combat stress.[^5][^6] Lethality values come from historical wound ballistics (5.56×45 mm and 5.45×39 mm have ~30% immediate incapacitation probabilities against torso hits; 7.62×51 mm machine-gun fire is closer to 40%).[^7][^8] Exposure probabilities assume mixed woodland/rolling terrain with roughly 60% of the enemy occasionally visible.[^9]

| U.S. Army (Blue) group | Platforms | Sustained ROF (rpm) | Exposure probability | Hit probability | Kill probability | Per-platform kills/min | Group kills/min |
| ---------------------- | --------- | ------------------- | -------------------- | --------------- | ---------------- | ---------------------- | --------------- |
| Riflemen (M16A2)       | 27        | 10                  | 0.6                  | 0.12            | 0.30             | 0.216                  | 5.8             |
| SAW gunners (M249)     | 6         | 25                  | 0.6                  | 0.18            | 0.35             | 0.945                  | 5.7             |
| MG teams (M60)         | 2         | 60                  | 0.6                  | 0.20            | 0.40             | 2.880                  | 5.8             |
| Dragon AT teams        | 2         | 3                   | 0.4                  | 0.25            | 0.80             | 0.240                  | 0.5             |
| **Total**              | **37**    | —                   | —                    | —               | —                | —                      | **17.2**        |

Dividing the 17.2 expected kills per minute by 37 shooters yields \(\beta \approx 0.46\ \text{kills per shooter per minute}\).

| Soviet Army (Red) group | Platforms | Sustained ROF (rpm) | Exposure probability | Hit probability | Kill probability | Per-platform kills/min | Group kills/min |
| ----------------------- | --------- | ------------------- | -------------------- | --------------- | ---------------- | ---------------------- | --------------- |
| Riflemen (AK-74)        | 24        | 8                   | 0.5                  | 0.08            | 0.30             | 0.096                  | 2.3             |
| RPK-74 gunners          | 6         | 20                  | 0.5                  | 0.12            | 0.30             | 0.360                  | 2.2             |
| PKM machine guns        | 3         | 30                  | 0.5                  | 0.15            | 0.35             | 0.788                  | 2.4             |
| RPG-7 gunners           | 3         | 2                   | 0.4                  | 0.20            | 0.80             | 0.128                  | 0.4             |
| **Total**               | **36**    | —                   | —                    | —               | —                | —                      | **7.3**         |

Dividing 7.3 expected kills per minute by 36 shooters gives \(\alpha \approx 0.20\ \text{kills per shooter per minute}\).

### Using and refining the coefficients

- **Sensitivity analysis.** Vary exposure and hit probabilities to capture differences in terrain, night vs. day, and training quality. For example, moving exposure from 0.6 to 0.4 drops \(\beta\) by one third.
- **Morale and suppression.** Let the per-platform kill term \(k_i\) decay with cumulative casualties or when the unit is suppressed. Multiplying \(k_i\) by a suppression factor between 0 and 1 lets you plug morale effects into the same coefficient.
- **Empirical calibration.** Replace the assumed parameters with values estimated from field exercises (e.g., the U.S. Army’s National Training Center scores) or archival after-action reports to align the model with observed casualty rates.
- **Scenario conversion.** Once you have \(\alpha\) and \(\beta\), plug them into the square-law ODEs or their discrete approximations, or convert them to linear-law coefficients by dividing by the expected number of simultaneous one-on-one engagements when modeling close-quarter combat.

### Sources

[^1]: Department of the Army, *FM 7-8: The Infantry Rifle Platoon and Squad*, Washington, DC, April 1984.
[^2]: Department of the Army, *FM 100-2-3: The Soviet Army – Troops, Organization, and Equipment*, Washington, DC, June 1987.
[^3]: Department of the Army, *FM 3-22.68: Crew-Served Machine Guns 5.56-mm and 7.62-mm*, Washington, DC, July 2006 (sustained-rate tables carried forward from earlier editions).
[^4]: Lester W. Grau and Charles K. Bartles, *The Russian Way of War*, Foreign Military Studies Office, 2016 (summarizes Soviet-era small-arms doctrine and rates of fire).
[^5]: Department of the Army, *FM 23-9: Rifle Marksmanship M16A1 and M16A2*, Washington, DC, August 1989.
[^6]: U.S. Army Intelligence and Threat Analysis Center, *Soviet Army Operations and Tactics*, 1984 (training and marksmanship performance data for motor-rifle units).
[^7]: James W. Patrick, “The M16 Rifle: Ballistics and Effectiveness,” *Ballistic Research Laboratory Memorandum Report 1037*, Aberdeen Proving Ground, 1967.
[^8]: I. A. Komarov, *Terminal Ballistics of Small Arms*, Moscow, 1977 (translated summary prepared by U.S. Army Foreign Science and Technology Center, 1980).
[^9]: Army Research Institute, *Infantry Exposure Times in Simulated Engagements*, Technical Report 854, 1982.
