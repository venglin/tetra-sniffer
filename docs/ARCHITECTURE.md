# Architecture

```
HackRF (cs8 IQ) ──► ChannelFilter ──► Demodulator ──► LowerMac ──► UpperMac ──► Printer
 hackrf.py          dsp.py              demod.py       lower_mac.py  lower_mac.py  receiver.py
 or a file          shift + resample    π/4-DQPSK      coding.py     mac.py        + gsmtap.py
                    to 144 kHz          → soft bursts  → blocks      → PDUs

activity.py: AACH + MAC headers → slot occupancy, traffic sessions, signalling load
scanner.py: survey (PSD) → candidates → verify (MultiReceiver per window)
sim.py:     synthetic downlink generator used by tests, `selftest` and the example
```

| module | contents |
|---|---|
| `hackrf.py` | minimal ctypes binding to libhackrf (streaming RX into a queue) |
| `dsp.py` | RRC filter, π/4-DQPSK modulator, streaming channel filter, PSD |
| `burst.py` | training sequences, continuous downlink burst layouts, dibit mapping |
| `demod.py` | carrier offset acquisition, burst search and tracking, soft bits |
| `coherent.py` | carrier phase tracking and max-log BCJR (coherent soft bits) |
| `coding.py` | CRC-16, RCPC encoder + batch Viterbi, interleaver, scrambler, RM(30,14) |
| `lower_mac.py` | burst → logical channels, cell state / TDMA time, shared colour codes, fragment reassembly |
| `mac.py` | MAC / LLC / MLE / MM / CMCE parsers and one-line descriptions |
| `receiver.py` | per-channel and multi-channel receivers, output filtering |
| `activity.py` | cell activity statistics (`--report`) |
| `gsmtap.py` | GSMTAP packets, UDP sender, PCAP writer |
| `scanner.py` | band survey, peak detection, capture planning, verification |
| `sim.py` | synthetic TETRA main carrier, impairments, cs8 writer |
| `cli.py` | `scan`, `live`, `record`, `file`, `selftest` |

## Air interface summary

- π/4-DQPSK, 18 ksymbol/s (36 kbit/s), root-raised-cosine roll-off 0.35, 25 kHz raster
  (carrier offsets 0, ±6.25 or +12.5 kHz).
- TDMA: a timeslot is 255 symbols (510 bits, 14.167 ms), 4 timeslots form a frame,
  18 frames a multiframe and 60 multiframes a hyperframe.
- The downlink of a base station is continuous: bursts follow each other without guard
  time. Each burst begins with q11..q22 and ends with q1..q10, so the 22-bit training
  sequence q straddles every slot boundary.

Bit layout of the two continuous downlink bursts (0-based bit offsets):

```
normal (NDB):  q11-22 [0,12) | ha [12,14) | block 1 [14,230) | bb1 [230,244) | n|p [244,266)
               | bb2 [266,282) | block 2 [282,498) | hb [498,500) | q1-10 [500,510)
sync (SB):     q11-22 [0,12) | hc [12,14) | freq. corr. [14,94) | BSCH [94,214) | y [214,252)
               | bb [252,282) | block 2 [282,498) | hd [498,500) | q1-10 [500,510)
```

Training sequence p means two half-slot logical channels. Training sequence n means one
full-slot channel (as in osmo-tetra, which matches what is seen on air). `bb` is the
broadcast block, which carries the AACH.

## Channel filter (`dsp.ChannelFilter`)

The filter shifts each carrier to 0 Hz and resamples it to 144 kHz (8 samples/symbol).

- For integer decimation (the live rate of 2.304 Msps gives D = 16) it is a
  **frequency-translating FIR**. The taps are modulated, so the only remaining rotation
  runs at the output rate, which makes it about 4× faster than mixing at the input rate.
- Rational ratios (for example recordings at 2.4 Msps) use `upfirdn` with up/down.
- Streaming is exact: the buffer start is always kept at a multiple of `down`, so the
  output does not depend on how the input is chunked. A test checks this.
- The filter passes ±13 kHz and stops at f_out − 13 kHz, where the first alias of the
  channel would land. Adjacent channels are removed later by the RRC matched filter.

## Demodulator (`demod.Demodulator`)

1. **Carrier offset acquisition** (on the first 150 ms):
   - *Coarse:* correlate the PSD (140 Hz bins) with the raised-cosine spectrum of an
     18 ksymbol/s signal and take the peak within ±10 kHz. A shape match is not pulled by
     adjacent channels the way a power centroid would be.
   - *Fine:* for differential symbols d_k = r_k·r*_{k−1}, the values of d⁴ cluster at −1,
     so `angle(−Σ d⁴/|d|³)/4` gives the residual (range ±2.25 kHz). It is evaluated at the
     best of the 8 sampling phases.
2. **Burst search.** On 8 slots of RRC-filtered samples, compute the normalised
   correlation of the differential signal with the q template:
   `|Σ d[n+8k]·conj(t_k)| / Σ|d[n+8k]|`. The result is insensitive to carrier phase and
   (small) frequency offsets. It is averaged modulo 2040 samples (one slot), and a peak
   above 0.55 gives the slot timing.
3. **Tracking.** For each expected burst, the score over ±4 samples is the mean of three
   correlations: q before the burst, q after it, and the best of n / p / y in the middle.
   The best offset is refined by parabolic interpolation and fed into a first-order timing
   loop (gain 0.3). This follows the HackRF clock error, which is ±0.04 samples per slot
   at 20 ppm. The middle sequence also gives the burst type. Eight consecutive bad
   bursts send the demodulator back to search.
4. **Symbols.** Symbols −1…254 are linearly interpolated at the refined instants. The
   per-burst residual offset (4th-power estimate on the differential symbols) is fed back
   to the mixer with gain 0.05. The differential symbols also give the MER shown in the
   status line.
5. **Soft bits**, from one of two detectors (positive soft value = bit 0):
   - *Differential* (`--demod differential`). Table 5.1 maps the dibits (00, 01, 11, 10)
     to +π/4, +3π/4, −3π/4 and −π/4. So after removing the per-burst residual offset,
     the first bit is the sign of Im(d) and the second the sign of Re(d), normalised by
     the mean |d|.
   - *Coherent* (default, `coherent.py`), described below.

## Coherent detection (`coherent.py`)

Every phase change is π/4 + d·π/2 with d = 0, 1, 2, 3 for dibits 00, 01, 11, 10.
Multiplying symbol k by e^(−jkπ/4) removes the deterministic part and leaves QPSK, whose
constellation index evolves as m_k = m_(k−1) + d_k (mod 4).

1. **Carrier phase.** The 4th power of the de-rotated symbols removes the data. It is
   averaged over a 33-symbol sliding window and unwrapped, giving the carrier phase
   modulo π/2. The π/2 ambiguity is harmless, because the information is in the changes
   of m.
2. **No per-burst offset correction.** The per-burst frequency estimate is deliberately
   *not* applied here. At low SNR it is noisy, and a wrong phase ramp across 255 symbols
   hurts a coherent detector, especially at the burst edges. The slowly converging
   mixer loop plus the sliding-window tracker work better: applying the per-burst
   estimate cost about 2 dB in tests.
3. **Max-log BCJR.** A forward-backward pass over the 4 states (uniform initial state,
   branch metric (2/σ²)·Re(u_k e^(−jmπ/2)), σ² estimated per burst) gives max-log LLRs
   for both bits of every dibit.

It is batched over all bursts of an input chunk (about 1.4 ms per burst). On the AWGN
symbol level its BER is roughly 2 dB better than differential detection, and the
`selftest` table in the README shows a similar gain for decoded blocks.

## Lower MAC (`lower_mac.LowerMac`, `coding.py`)

Coding chain, per logical channel:

```
type-1 ─► +CRC-16 +4 tail ─► type-2 ─► RCPC (K=5, rate 1/4 mother, punctured to 2/3) ─► type-3
       ─► block interleaver (K, a) ─► type-4 ─► scrambler (extended colour code) ─► type-5
```

| channel | type-1 | type-5 | interleaver a | carried in |
|---|---|---|---|---|
| BSCH | 60 | 120 | 11 | SB, scrambled with colour code 0 |
| SCH/HD, BNCH, STCH | 124 | 216 | 101 | SB block 2, NDB(p) blocks 1 and 2 |
| SCH/F | 268 | 432 | 103 | NDB(n), blocks 1+2 |
| AACH | 14 | 30 | – (RM(30,14)) | broadcast block of every burst |

- The **Viterbi decoder** is vectorised over a batch of blocks. All blocks of the same
  kind from one input chunk are decoded together (16 states, numpy, about 0.1 ms per
  block).
- **CRC checking** uses the affine-map form of the CRC (one matrix multiply per batch).
  The good residue is 0x1D0F.
- **AACH** is decoded by a maximum-likelihood search over all 16384 RM(30,14) codewords
  and accepted with ≤ 3 disagreeing hard bits (minimum distance 8).
- **Order of decoding.** Synchronisation bursts are decoded first. A BSCH that passes its
  CRC gives MCC, MNC and the colour code, which set the scrambling code of every other
  block on the carrier. It also anchors the TDMA time (TN/FN/MN), which is then advanced
  using the demodulator's slot counter. Blocks received earlier in the same batch are
  decoded as well.
- A full-slot burst is skipped as **TCH** when the AACH says the slot carries traffic,
  or when its SCH/F CRC fails.
- Only blocks whose first four bits are `1000` (MAC broadcast, SYSINFO) are labelled
  **BNCH**. Wireshark always dissects a BNCH as SYSINFO, so ACCESS-DEFINE is labelled
  SCH/HD.

### Colour codes for secondary carriers (`CellRegistry`)

All channels of one `MultiReceiver` share a `CellRegistry`. A channel that decodes a
BSCH registers its colour code together with the start sample and TDMA time of that
burst. A channel without its own BSCH tests every registered code on up to 8 bursts. It
accepts a code when a half- or full-slot block passes its CRC, or when at least 90 % of
the AACHs decode with ≤ 3 errors; a wrong code passes an AACH only about 7 % of the time.

If the channel's bursts start a whole number of slots (±2 %) away from the registering
burst, the channel also inherits the TDMA time. The carriers of one base station are
slot-aligned, as observed on air. Codes given with `--cell` have no timing, so those
channels decode with an unknown TDMA time.

## Upper MAC (`lower_mac.UpperMac`, `mac.py`)

`parse_mac_block` walks the MAC PDUs in a block, following MAC PDU association (several
PDUs per block, length indication, fill-bit removal, Null PDU). MAC-RESOURCE handling:

- **Addresses.** It decodes all address types.
- **Channel allocation element.** Decoded when the PDU is not encrypted; with encryption
  the element itself is encrypted.
- **TM-SDU.** Passed on to the LLC / MLE parser.
- **Fragments.** MAC-RESOURCE (length indication 0x3F), MAC-FRAG and MAC-END are
  reassembled per timeslot. A MAC-END inherits the encryption mode of the MAC-RESOURCE
  that started the fragment: its channel allocation element is decoded only when that
  fragment is clear, is marked encrypted when it is encrypted, and is marked unknown
  when the first fragment was not received. PDUs that follow a MAC-END in the same block
  are parsed as well.

Decoded beyond the PDU type:

| PDU | fields |
|---|---|
| ACCESS-DEFINE (MAC) | access code, IMM, WT, Nu, minimum priority, subscriber class or GSSI |
| D-NWRK-BROADCAST (MLE) | cell reselection parameters, service level, TETRA network time (UTC, local offset, year), neighbour cells (CA, carrier, optional carrier extension, MCC, MNC, LA, …) |
| D-SETUP | call id, basic service, grant, priority, notification, temporary address, calling party |
| D-CONNECT | call id, grant, ownership, priority, basic service, temporary address, notification |
| D-TX-GRANTED | call id, grant, encryption control, notification, transmitting party |
| D-RELEASE / D-DISCONNECT | call id, cause (+ notification) |
| D-STATUS, D-SDS-DATA | calling party, status value / SDS length and protocol id (never the content) |

Type-2 elements follow the O-bit / P-bit rules. Type-3 elements are not parsed.

### Cross-check with Wireshark

A PCAP written with `--pcap` was dissected by Wireshark's TETRA dissector:

- **Matching values:** SYSINFO, MAC-RESOURCE, MAC-END, LLC, D-SETUP (with calling
  party), D-CONNECT, D-TX-GRANTED (with transmitting party), D-STATUS, D-SDS-DATA and
  ACCESS-DEFINE.
- **ACCESS-DEFINE GSSI.** Wireshark's ASN.1 gives the GSSI the range 0…2²⁵−1, so it reads
  25 bits. TETRA addresses are 24 bits, as used here.
- **D-NWRK-BROADCAST.** Wireshark's ASN.1 (`D-NWRK-BRDADCAST`) has a `pdu-type` field
  *in addition to* the MLE PDU type that selects it, so it reads the PDU type twice and
  every later field is shifted by 3 bits. Its CMCE PDUs do not do this, and they match
  tetrascan. tetrascan reads the PDU type once, as for every other PDU.
- **Neighbour cells.** The neighbour cell information element is defined neither in
  Wireshark nor in osmo-tetra, so its layout (clause 18.5.17) has not been cross-checked.
- **SCH/HD.** Wireshark's GSMTAP handler maps sub-type SCH/HD to nothing and drops those
  blocks. Signalling in half slots (SCH/HD, and BNCH other than SYSINFO) therefore shows
  only in tetrascan's own output.

## Activity report (`activity.py`)

Every `ChannelReceiver` feeds its blocks into an `Activity` object.

- **Slot occupancy.** For each AACH, outside frame 18, the downlink usage of the slot
  (unallocated, assigned control, common control or traffic) and the uplink usage (header
  3: traffic or unallocated, otherwise control) are counted per timeslot. The share of
  synchronisation bursts per slot is counted as well.
- **Traffic sessions.** A session opens when a slot shows a traffic usage marker, and is
  extended while the same marker keeps appearing with gaps of at most 3 s. It ends when
  the slot shows something else, the marker changes, or the gap is longer. Frame 18
  neither extends nor ends a session. A session still open at the end of a reporting
  period is reported as ongoing and carried over to the next period.
- **Signalling load.** Counts of null and non-null MAC-RESOURCE PDUs, encrypted vs clear,
  distinct addresses (only their number is reported), fragments, random-access
  acknowledgements, channel allocations, and the types of clear layer-3 PDUs.

Nothing in the report depends on decryption: the AACH and the MAC header fields are never
encrypted.

## Scanner (`scanner.py`)

1. **Survey.** The HackRF runs at 20 Msps with LO steps of 7.5 MHz. Only offsets
   +0.25…+7.75 MHz of each step are used, which keeps the DC spike and the filter edges
   out of the data. A Welch PSD with 2.4 kHz bins is taken at every step, and 380–470 MHz
   takes about one second.
2. **Candidates.**
   - Power in ±9 kHz is compared with a 25th-percentile noise floor over 1 MHz.
   - Local maxima above the threshold are snapped to the 6.25 kHz raster.
   - A −10 dB width of 15–30 kHz marks a carrier as TETRA-like. TETRA occupies about
     22 kHz; DMR and analogue FM are narrower, LTE and the like much wider.
3. **Verification.** Candidates are grouped into windows of ≤ 1.6 MHz. The LO is chosen
   so that every carrier is 40 kHz–0.9 MHz away from it. Each window is captured at
   2.304 Msps and decoded by `MultiReceiver`.
4. **Second pass.** The raw captures are kept. Every carrier that locked without finding
   a colour code is decoded again, with all colour codes learned from BSCHs in any window
   (secondary carriers can be more than 1.6 MHz away from their main carrier).
5. **Clock error.** The scanner compares the measured frequency of each main carrier with
   the exact frequency in its own SYSINFO.

## Known limitations and ideas

- **Better phase tracking.** Using the known training sequences (q, n/p, y) as pilots, or
  iterating with the decoder, could gain more at very low SNR.
- **Colour code from a lone weak carrier.** A carrier on its own still needs either one
  good BSCH or `--cell`. Its colour code could be searched (MCC/MNC from a neighbour list,
  64 colour codes) with the CRC as the test.
- **Speed.** The BCJR and Viterbi loops are numpy-vectorised over bursts but iterate over
  symbols in Python. A compiled kernel would make many-carrier live decoding cheaper.
- **Not decoded yet.** Type-3 elements of L3 PDUs, and most MM PDU contents.
