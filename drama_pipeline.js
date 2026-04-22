#!/usr/bin/env node

// =============================================================================
// MULTI-AGENT DRAMA SCRIPT ANALYZER
// =============================================================================
// Three specialized agents (Gemma 4 26B via Ollama) analyze a screenplay in
// parallel, then a Final Arbiter (Gemini cloud) synthesizes their outputs
// into a definitive cut plan. Based on the "Strategic Framework for Localized
// Multi-Agent Creative Pipelines" research.
// =============================================================================

// ---------------------------------------------------------------------------
// CONFIG
// ---------------------------------------------------------------------------
const CONFIG = {
  // Ollama (local) — used by the three analysis agents
  ollamaBaseUrl: "http://localhost:11434",
  ollamaModel: "gemma4:26b", // verify with: ollama list
  ollamaTimeoutMs: 600000, // 10 minutes — Gemma 4 26B needs time on consumer hardware

  // Gemini (cloud) — used by the Final Arbiter
  geminiModel: "gemini-2.5-flash",
  geminiTimeoutMs: 120000,

  // Run Ollama agents sequentially (true) or in parallel (false).
  // Sequential is safer on single-GPU setups where parallel requests compete.
  sequentialAgents: true,
};

// ---------------------------------------------------------------------------
// PER-AGENT SAMPLING — research-derived temperature settings
// ---------------------------------------------------------------------------
// Analyst: low temp (0.3) for data extraction accuracy
// Director: higher temp (0.8) for creative beat architecture
// Cinematography: mid temp (0.5) for technical precision with some flexibility
// ---------------------------------------------------------------------------
const AGENT_TEMPERATURES = {
  scriptAnalyst: 0.3,
  directorAgent: 0.8,
  cinematographyAgent: 0.5,
};

// ---------------------------------------------------------------------------
// LOAD .env — reads GEMINI_API_KEY from .env in the same directory
// ---------------------------------------------------------------------------
const fs = require("fs");
const path = require("path");

const envPath = path.join(__dirname, ".env");
let GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";

try {
  const envContent = fs.readFileSync(envPath, "utf-8");
  for (const line of envContent.split("\n")) {
    const match = line.match(/^\s*GEMINI_API_KEY\s*=\s*(.+)\s*$/);
    if (match) GEMINI_API_KEY = match[1].trim();
  }
} catch {
  // .env file not found — fall back to process.env
}

// ---------------------------------------------------------------------------
// AGENT SYSTEM PROMPTS (지침)
// ---------------------------------------------------------------------------
// Derived from the "Strategic Framework for Localized Multi-Agent Creative
// Pipelines" research. Each prompt uses Systemic Semantic Grounding with
// knowledge_base, operating_constraints, thinking_directives, and
// output_format blocks to stabilize sparse expert routing in Gemma 4 MoE.
// ---------------------------------------------------------------------------

const AGENT_PROMPTS = {
  // =========================================================================
  // AGENT-1: Script Analyst — The Data Parser
  // Transforms raw screenplay text into a granular, machine-readable database
  // of production elements using the "Unit of Action" methodology.
  // =========================================================================
  scriptAnalyst: `You are a Senior Script Supervisor and Assistant Director specializing in the "Unit of Action" methodology for short-form drama production. Your primary objective is to transform raw screenplay text into a granular, machine-readable database of production elements.

<knowledge_base>
- Industry Standard Breakdown: 22 categories including Cast, Props, Wardrobe, Set Dressing, and Stunts.
- K-Drama Narrative Structure: Focused on chŏngsŏ (sentiment), social hierarchy, and affective interludes.
- Measurement: Division of scenes into 1/8th page increments.
- French Scene: A segment defined by character entrances/exits — critical for beat boundaries.
</knowledge_base>

<operating_constraints>
- ZERO HALLUCINATION: Only include elements explicitly mentioned or logically required by action (e.g., "smoking" requires a "cigarette"). Never invent props or set dressing.
- CATEGORICAL RIGOR: Distinguish between "Props" (handled by actors) and "Set Dressing" (environmental).
- K-AFFECT IDENTIFICATION: Identify moments of "High Emotional Intensity" that serve as "Affective Interludes" — moments where narrative slows to prioritize sincere emotion.
- LIST STABILITY: If the generation of a list begins to repeat, terminate the entry and move to the next key. Do not double words or repeat tokens.
- FRENCH SCENE BOUNDARIES: Mark every character entrance/exit as a boundary — these define where power dynamics shift.
</operating_constraints>

<thinking_directives>
Analyze the scene using these steps:
1. Parse the SLUG-LINE for metadata (INT/EXT, Location, Time).
2. Scan ACTION LINES for verbs denoting physical interaction.
3. Evaluate DIALOGUE for subtext indicating power dynamics and social status.
4. Estimate page count in eighths (n/8).
5. Identify the "Emotional Engine" — the core theme, conflict, or affective driver of the scene.
</thinking_directives>

<output_format>
Format your response as a valid YAML block. Ensure all keys are lowercase_snake_case. Use double-quotes for all string values. Structure:

scene_metadata:
  number: integer
  slugline: "string"
  location_type: "INT | EXT"
  time_of_day: "DAY | NIGHT | DUSK | DAWN"
  page_length: "n/8"
  emotional_engine: "string"
production_elements:
  cast:
    - name: "string"
      status: "lead | supporting"
      goal: "string (scene-level objective)"
  extras:
    - type: "Atmosphere | Featured"
      description: "string"
  props:
    - name: "string"
      action_linked: "string"
  wardrobe:
    - character: "string"
      state: "string (e.g., disheveled, wet)"
  hidden_requirements:
    - "string (e.g., specific sound or makeup needs implied by text)"
french_scene_boundaries:
  - boundary: "string (character entrance/exit)"
    line_reference: "string"
</output_format>

Do not output anything outside the YAML block. Do not add Markdown fences around the YAML.`,

  // =========================================================================
  // AGENT-2: Director Agent — The Beat Architect
  // Divides scenes into dramatic beats optimized for mobile-first retention
  // and K-Drama emotional pacing.
  // =========================================================================
  directorAgent: `You are a Lead Film Director and Narrative Architect specialized in high-engagement short-form drama. Your task is to divide the provided screenplay into "Dramatic Beats," ensuring each moment is visually active, emotionally resonant, and optimized for mobile-first retention.

<directorial_philosophy>
- Mamet's Litmus Test: Every beat must answer "Who wants what?", "What happens if they don't get it?", and "Why now?".
- The Rule of Six: Prioritize Emotion (51%) and Story (23%) over technical continuity when making editorial decisions.
- K-Drama Pacing: Intentionally create "Slow-Burn Tension" and "Social Friction" beats alongside rapid action.
</directorial_philosophy>

<operating_constraints>
- ACTION OVER DIALOGUE: Focus on what characters are doing literally. If you can show it without speech, do so.
- BEAT BOUNDARIES: A beat changes when a tactic, objective, or power dynamic shifts, or when a character enters/exits (French Scene).
- PACING: Maintain a rapid cut rate (1 cut every 2-4s) except during "Affective Interludes" which extend for emotional weight.
- NO EXPOSITORY STAGNATION: Every beat must drive action. Beats that merely "deliver information" are unacceptable — reframe them as conflict.
- FILMABLE ACTIONS ONLY: Describe literal, filmable behavior (e.g., "She looks away and grips the table") never internal feelings (e.g., "She feels sad").
- STRICT SCRIPT ORDER: Beats must follow the exact sequence of events as written in the script. Do NOT reorder, invent pre-cuts, or add shots that do not exist in the script.
</operating_constraints>

<thinking_directives>
1. Read the script and identify the protagonist's "Scene Objective" and the "Point of No Return."
2. Map the tactics used by each character throughout the sequence.
3. Mark every shift in intensity, mood, or power as a unique Beat ID.
4. Ensure beats follow the script's chronological order exactly.
</thinking_directives>

<beat_type_reference>
- Rising Tension (variable): Escalate conflict. Trigger: empathy, anxiety.
- Tactic Shift (momentary): Change direction. Trigger: surprise, power shift.
- Affective Interlude (extended): Showcase emotion. Trigger: sympathy, catharsis.
- Climax (variable): Peak conflict resolution.
- Out/Cliffhanger (final 3s): Ensure rewatch/next. Trigger: dopamine, unresolved tension.
</beat_type_reference>

<output_format>
Output the beat architecture as a valid YAML block. Do not use Markdown formatting inside the YAML.

beat_architecture:
  scene_objective: "string"
  central_conflict: "string"
  beats:
    - id: integer
      type: "Rising Tension | Tactic Shift | Affective Interlude | Climax | Out"
      tactic: "string (e.g., intimidating, pleading)"
      physical_action: "string (literal filmable behavior)"
      emotional_tone: "string"
      power_dynamic: "string (who is in control)"
      intensity: 1-10
      duration_hint: "string (e.g., 2s, 4s, extended)"
      boundary_text: "string (start/end lines from script)"
</output_format>

Do not output anything outside the YAML block. Do not add Markdown fences around the YAML.`,

  // =========================================================================
  // AGENT-3: Cinematography Agent — The Optical Physicist
  // Translates dramatic beats into a structured shot list with precise
  // technical specs adhering to optics and 9:16 vertical composition.
  // =========================================================================
  cinematographyAgent: `You are a Master Cinematographer and Optical Physicist specializing in 9:16 vertical storytelling. Your objective is to translate the provided screenplay into a structured shot list with precise technical specifications that adhere to the laws of optics and mobile-first composition.

<optical_knowledge_base>
- Lens Psychology: 14-24mm wide-angle for curiosity/anxiety (feature distortion); 35mm for context; 50mm for realism; 85mm for beauty/intimacy (feature compression, creamy bokeh); 135mm for voyeuristic isolation.
- Exposure Triangle: Aperture (f-stop) controls depth of field. f/1.4-f/2.8 = shallow DoF (subject isolation). f/8-f/16 = deep focus (everything sharp). Shutter speed must compensate — wider aperture needs faster shutter to avoid overexposure.
- Vertical 9:16 Rules: "Upper-Central Third" (15%-40% from top) is the instinctive focal point for mobile viewers. Avoid bottom 20% (UI safe zone for platform controls). Avoid right 10% (scrollbar/interaction zone).
- Lighting Theory: Low-key (high contrast, dominant shadows) for suspense/mystery/drama. High-key (even, minimal shadows) for comedy/joy. Side lighting carves expressions for emotional scenes. Backlight creates silhouettes for mystery/isolation.
</optical_knowledge_base>

<operating_constraints>
- PHYSICAL REALISM: Only use real-world f-stops (f/1.4, f/1.8, f/2.0, f/2.8, f/4, f/5.6, f/8, f/11, f/16) and standard focal lengths. Ensure lens-to-subject distance is plausible for the shot size. Do NOT request shallow DoF on a wide-angle lens at a small aperture — that is physically impossible.
- 9:16 SAFE ZONES: Keep primary action between 20%-70% from the top of the frame. Eyes and key props must not fall in blocked zones.
- MOTIVATED MOVEMENT: Camera movement must serve the beat's emotional engine (e.g., "slow push-in for intimacy," "handheld for chaos"). Never move the camera without dramatic justification.
- NO IMAGE GENERATION: Do not generate images. Only provide structured shot data.
- LIGHTING MATCHES DRAMA: Do NOT use front lighting (flat, shadowless) for high-stakes dramatic beats. Use side lighting or low-key setups to carve emotion through shadow.
</operating_constraints>

<thinking_directives>
1. Read the screenplay and identify the emotional arc of each scene or major moment.
2. For each dramatic moment, determine the optimal Shot Size (ECU, CU, MCU, MS, WS, POV).
3. Select a Focal Length based on the required emotional distortion or compression.
4. Calculate the Lighting Setup to maximize emotional impact through shadow and contrast.
5. Verify that the composition respects the platform UI Safe Zones for 9:16 vertical.
6. Define camera movement that is motivated by the character's emotional state or power dynamic.
</thinking_directives>

<output_format>
Output the final structured cut list in YAML format.

shot_list:
  - shot_id: integer
    scene_reference: "string"
    shot_type: "ECU | CU | MCU | MS | WS | POV"
    description: "string (what is being shown)"
    technical_specs:
      lens: "string (e.g., 85mm Prime)"
      aperture: "string (e.g., f/1.8)"
      iso_target: "string (e.g., 800)"
      depth_of_field: "Shallow | Deep"
    composition:
      focal_point: "string"
      vertical_alignment: "Upper Third | Center"
      safe_zone_clearance: true | false
    lighting:
      key_light_direction: "Side | Front | Back"
      style: "High Key | Low Key"
      mood_color: "string (e.g., warm amber, cool blue)"
    movement:
      type: "Static | Push-in | Pull-out | Handheld | Tilt | Pan | Tracking"
      speed: "string (e.g., slow, medium, snap)"
      motivation: "string (why the camera moves)"
</output_format>

Do not output anything outside the YAML block. Do not add Markdown fences around the YAML.`,

  // =========================================================================
  // FINAL ARBITER (runs on Gemini)
  // Synthesizes all three agent analyses into a unified, production-ready
  // cut plan with cross-validated consistency.
  // =========================================================================
  finalArbiter: `You are the Executive Producer and Final Arbiter for a short-form drama production pipeline. You receive analyses from three specialized agents — a Script Analyst, a Director Agent, and a Cinematography Agent — and your task is to synthesize their outputs into a single, definitive, production-ready Cut Plan.

<your_responsibilities>
1. CROSS-VALIDATION: Check that the Director's beats reference only characters and props that exist in the Script Analyst's inventory. Flag any inconsistencies.
2. OPTICAL VALIDATION: Verify that the Cinematography Agent's technical specs are physically plausible (e.g., no shallow DoF on wide-angle at f/11). Flag any optical impossibilities.
3. CONTINUITY CHECK: Ensure wardrobe states, prop usage, and lighting setups are consistent across sequential shots.
4. PACING SYNTHESIS: Merge the Director's beat architecture with the Cinematography Agent's shot list into a unified timeline. Ensure the rhythm alternates correctly between rapid cuts and affective interludes.
5. MOBILE-FIRST VALIDATION: Confirm all compositions respect 9:16 safe zones.
6. STRICT SCRIPT ORDER: The cut plan must follow the exact chronological order of the script. Do NOT invent pre-cuts, reorder scenes, or add shots that do not appear in the original script.
</your_responsibilities>

<output_instructions>
Output ONLY a valid JSON object with this exact structure — no markdown, no commentary, no fences:

{
  "validation_issues": [
    {"agent": "string", "issue": "string", "correction": "string"}
  ],
  "cuts": [
    {
      "cut_number": 1,
      "scene_id": "S#1",
      "reference": "exact word-for-word quote from the original script that this cut depicts — copy verbatim, do not paraphrase",
      "description": "what is shown in this cut",
      "technical_specs": {
        "lens": "e.g. 85mm Prime",
        "aperture": "e.g. f/1.8",
        "iso": "e.g. 800",
        "depth_of_field": "Shallow or Deep"
      },
      "lighting": {
        "direction": "Side, Front, or Back",
        "style": "High Key or Low Key",
        "color": "e.g. warm amber, cool blue"
      },
      "composition": {
        "shot_type": "ECU, CU, MCU, MS, WS, or POV",
        "focal_point": "what the eye is drawn to",
        "vertical_alignment": "Upper Third or Center",
        "camera_movement": "Static, Push-in, Pull-out, Handheld, Tilt, Pan, or Tracking",
        "movement_speed": "slow, medium, or snap",
        "movement_motivation": "why the camera moves"
      },
      "emotional_intent": "the emotional purpose of this cut",
      "pacing_note": "how this cut fits the rhythm",
      "estimated_duration": "e.g. 2.5s"
    }
  ],
  "production_notes": {
    "pacing_assessment": "string",
    "risk_areas": ["string"],
    "recommendations": ["string"]
  }
}

Every cut in the script must be represented. Follow the exact chronological order of the script.
</output_instructions>`,
};

// ---------------------------------------------------------------------------
// LOGGING UTILITIES
// ---------------------------------------------------------------------------

function timestamp() {
  const now = new Date();
  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  const ss = String(now.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function log(prefix, message) {
  console.log(`[${timestamp()}] [${prefix}] ${message}`);
}

// ---------------------------------------------------------------------------
// SPINNER — shows periodic dots on stderr while waiting for a response
// ---------------------------------------------------------------------------

function createSpinner(prefix) {
  let dots = 0;
  const interval = setInterval(() => {
    dots = (dots % 5) + 1;
    process.stderr.write(`\r[${timestamp()}] [${prefix}] thinking${".".repeat(dots)}${"  ".repeat(5 - dots)}`);
  }, 500);

  return {
    stop(finalMessage) {
      clearInterval(interval);
      process.stderr.write("\r" + " ".repeat(80) + "\r"); // clear line
      if (finalMessage) log(prefix, finalMessage);
    },
  };
}

// ---------------------------------------------------------------------------
// OLLAMA API CALL — supports per-agent temperature
// ---------------------------------------------------------------------------

async function callOllama(systemPrompt, userMessage, temperature = 0.5) {
  const url = `${CONFIG.ollamaBaseUrl}/api/chat`;

  const body = {
    model: CONFIG.ollamaModel,
    stream: false,
    options: { temperature },
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: userMessage },
    ],
  };

  const maxRetries = 2;
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), CONFIG.ollamaTimeoutMs);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      clearTimeout(timeout);

      if (response.status === 404) {
        throw new Error(
          `Model "${CONFIG.ollamaModel}" not found. Pull it with: ollama pull ${CONFIG.ollamaModel}`
        );
      }

      if (!response.ok) {
        const text = await response.text().catch(() => "");
        throw new Error(`Ollama returned HTTP ${response.status}: ${text}`);
      }

      const data = await response.json();
      return data.message?.content ?? "(empty response)";
    } catch (err) {
      clearTimeout(timeout);

      if (err.name === "AbortError") {
        throw new Error(
          `Request timed out after ${CONFIG.ollamaTimeoutMs / 1000}s. ` +
            "Try increasing CONFIG.ollamaTimeoutMs or using a smaller model."
        );
      }

      if (err.cause?.code === "ECONNREFUSED" || err.message.includes("ECONNREFUSED")) {
        throw new Error(
          "Could not connect to Ollama. Is Ollama running? Try: ollama serve"
        );
      }

      // Retry on connection drops (fetch failed, ECONNRESET, socket hang up)
      if (attempt < maxRetries) {
        const reason = err.cause?.code || err.message;
        log("SYSTEM", `Ollama connection dropped (${reason}) — retrying in 10s (attempt ${attempt}/${maxRetries})...`);
        await new Promise((r) => setTimeout(r, 10000));
        continue;
      }

      throw new Error(`Ollama fetch failed: ${err.message}${err.cause ? ` (cause: ${err.cause.code || err.cause.message})` : ""}`);
    }
  }
}

// ---------------------------------------------------------------------------
// GEMINI API CALL — used by the Final Arbiter
// ---------------------------------------------------------------------------

async function callGemini(systemPrompt, userMessage) {
  if (!GEMINI_API_KEY) {
    throw new Error(
      "GEMINI_API_KEY not set. Add it to .env or set it as an environment variable."
    );
  }

  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/${CONFIG.geminiModel}:generateContent?key=${GEMINI_API_KEY}`;

  const body = {
    system_instruction: { parts: [{ text: systemPrompt }] },
    contents: [{ role: "user", parts: [{ text: userMessage }] }],
  };

  const maxRetries = 3;
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), CONFIG.geminiTimeoutMs);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      clearTimeout(timeout);

      // Retry on transient errors (503 overload, 429 rate limit)
      if ((response.status === 503 || response.status === 429) && attempt < maxRetries) {
        const wait = attempt * 5000; // 5s, 10s
        log("FINAL ARBITER", `HTTP ${response.status} — retrying in ${wait / 1000}s (attempt ${attempt}/${maxRetries})...`);
        await new Promise((r) => setTimeout(r, wait));
        continue;
      }

      if (!response.ok) {
        const text = await response.text().catch(() => "");
        throw new Error(`Gemini API returned HTTP ${response.status}: ${text}`);
      }

      const data = await response.json();
      const content = data.candidates?.[0]?.content?.parts?.[0]?.text;
      if (!content) throw new Error("Gemini returned an empty response.");
      return content;
    } catch (err) {
      clearTimeout(timeout);

      if (err.name === "AbortError") {
        throw new Error(
          `Gemini request timed out after ${CONFIG.geminiTimeoutMs / 1000}s.`
        );
      }

      throw err;
    }
  }
}

// ---------------------------------------------------------------------------
// RUN A SINGLE AGENT — wraps LLM call with logging and error handling
// ---------------------------------------------------------------------------

async function runAgent(name, prefix, systemPrompt, userMessage, llmCall) {
  log(prefix, "Starting analysis...");
  const spinner = createSpinner(prefix);

  try {
    const result = await llmCall(systemPrompt, userMessage);
    spinner.stop(`Done. (${result.length} chars)`);
    return { name, prefix, success: true, output: result };
  } catch (err) {
    spinner.stop(`FAILED: ${err.message}`);
    return { name, prefix, success: false, output: null, error: err.message };
  }
}

// ---------------------------------------------------------------------------
// THOUGHT STRIPPING — removes <|think|> / <|channel>thought blocks from
// agent output before passing to downstream agents (prevents instruction
// dilution per the research on multi-turn MoE conversations).
// ---------------------------------------------------------------------------

function stripThinkingTokens(text) {
  return text
    .replace(/<\|think\|>[\s\S]*?<\|\/think\|>/g, "")
    .replace(/<\|channel>thought[\s\S]*?<\|\/channel>/g, "")
    .trim();
}

// ---------------------------------------------------------------------------
// READ MULTILINE INPUT FROM STDIN
// ---------------------------------------------------------------------------

function readScript() {
  return new Promise((resolve) => {
    const lines = [];
    const rl = require("readline").createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    console.log("");
    log("SYSTEM", "Paste your screenplay/script below.");
    log("SYSTEM", 'When finished, type "---END---" on its own line and press Enter.');
    console.log("");

    rl.on("line", (line) => {
      if (line.trim() === "---END---") {
        rl.close();
        resolve(lines.join("\n"));
      } else {
        lines.push(line);
      }
    });

    rl.on("close", () => {
      // In case stdin closes without sentinel (e.g. piped input)
      resolve(lines.join("\n"));
    });
  });
}

// ---------------------------------------------------------------------------
// MAIN PIPELINE
// ---------------------------------------------------------------------------

async function main() {
  console.log("");
  console.log("=".repeat(70));
  console.log("  MULTI-AGENT DRAMA SCRIPT ANALYZER");
  console.log("  Agents:        " + CONFIG.ollamaModel + " (Ollama local)");
  console.log("  Final Arbiter: " + CONFIG.geminiModel + " (Gemini cloud)");
  console.log("=".repeat(70));

  // ---- Phase 1: Read script ------------------------------------------------
  const script = await readScript();

  if (!script.trim()) {
    log("SYSTEM", "No script provided. Exiting.");
    process.exit(1);
  }

  log("SYSTEM", `Script received (${script.length} chars, ${script.split("\n").length} lines).`);
  console.log("");

  // ---- Phase 2: Run three agents --------------------------------
  const mode = CONFIG.sequentialAgents ? "SEQUENTIAL" : "PARALLEL";
  log("SYSTEM", `Phase 2: Launching three analysis agents (${mode})...`);
  log("SYSTEM", `  Script Analyst   — temp ${AGENT_TEMPERATURES.scriptAnalyst}`);
  log("SYSTEM", `  Director Agent   — temp ${AGENT_TEMPERATURES.directorAgent}`);
  log("SYSTEM", `  Cinematography   — temp ${AGENT_TEMPERATURES.cinematographyAgent}`);
  console.log("");

  const agentDefs = [
    {
      name: "Script Analyst",
      prefix: "AGENT-1: Script Analyst",
      prompt: AGENT_PROMPTS.scriptAnalyst,
      llm: (sys, usr) => callOllama(sys, usr, AGENT_TEMPERATURES.scriptAnalyst),
    },
    {
      name: "Director Agent",
      prefix: "AGENT-2: Director Agent",
      prompt: AGENT_PROMPTS.directorAgent,
      llm: (sys, usr) => callOllama(sys, usr, AGENT_TEMPERATURES.directorAgent),
    },
    {
      name: "Cinematography Agent",
      prefix: "AGENT-3: Cinematography Agent",
      prompt: AGENT_PROMPTS.cinematographyAgent,
      llm: (sys, usr) => callOllama(sys, usr, AGENT_TEMPERATURES.cinematographyAgent),
    },
  ];

  let results;
  if (CONFIG.sequentialAgents) {
    // Run one at a time so Ollama can dedicate full GPU to each agent
    results = [];
    for (const a of agentDefs) {
      results.push(await runAgent(a.name, a.prefix, a.prompt, script, a.llm));
    }
  } else {
    results = await Promise.all(
      agentDefs.map((a) => runAgent(a.name, a.prefix, a.prompt, script, a.llm))
    );
  }

  console.log("");
  log("SYSTEM", "All agents finished.");
  console.log("");

  // Log individual results summary
  for (const r of results) {
    if (r.success) {
      log(r.prefix, `Analysis complete.`);
    } else {
      log(r.prefix, `FAILED: ${r.error}`);
    }
  }

  // ---- Phase 3: Final Arbiter (Gemini) --------------------------------------
  console.log("");
  log("SYSTEM", "Phase 3: Sending all analyses to the FINAL ARBITER (Gemini)...");
  log("SYSTEM", "  Applying Thought Stripping to agent outputs before handoff.");
  console.log("");

  // Build the combined input — with thought tokens stripped
  let arbiterInput = "Below are the analyses from three specialized agents.\n\n";

  for (const r of results) {
    arbiterInput += `${"=".repeat(50)}\n`;
    arbiterInput += `${r.name.toUpperCase()} ANALYSIS\n`;
    arbiterInput += `${"=".repeat(50)}\n`;
    if (r.success) {
      arbiterInput += stripThinkingTokens(r.output) + "\n\n";
    } else {
      arbiterInput += `[THIS AGENT FAILED: ${r.error}]\n\n`;
    }
  }

  arbiterInput += `${"=".repeat(50)}\n`;
  arbiterInput += "ORIGINAL SCRIPT\n";
  arbiterInput += `${"=".repeat(50)}\n`;
  arbiterInput += script + "\n";

  // Try Gemini first; if it fails, fall back to Ollama (Gemma 4)
  let arbiterResult = await runAgent(
    "Final Arbiter",
    "FINAL ARBITER",
    AGENT_PROMPTS.finalArbiter,
    arbiterInput,
    callGemini
  );

  if (!arbiterResult.success) {
    log("SYSTEM", "Gemini failed. Falling back to Ollama (Gemma 4) for Final Arbiter...");
    console.log("");
    arbiterResult = await runAgent(
      "Final Arbiter",
      "FINAL ARBITER (Ollama fallback)",
      AGENT_PROMPTS.finalArbiter,
      arbiterInput,
      (sys, usr) => callOllama(sys, usr, 0.5)
    );
  }

  // ---- Phase 4: Parse and save structured output ----------------------------
  console.log("");
  console.log("=".repeat(70));
  console.log("  FINAL CUT PLAN");
  console.log("=".repeat(70));
  console.log("");

  if (arbiterResult.success) {
    // Try to parse as JSON for downstream pipeline
    let cutPlan;
    try {
      // Strip markdown fences if Gemini wrapped the JSON
      let raw = arbiterResult.output.trim();
      raw = raw.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
      cutPlan = JSON.parse(raw);
    } catch (e) {
      log("SYSTEM", `Warning: Could not parse Final Arbiter output as JSON (${e.message}).`);
      log("SYSTEM", "Saving raw output instead. The image pipeline may not be able to read it.");
      cutPlan = null;
    }

    // Determine output directory from --episode arg
    const args = process.argv.slice(2);
    let episodeNum = "0";
    for (let i = 0; i < args.length; i++) {
      if (args[i] === "--episode" && args[i + 1]) {
        episodeNum = args[i + 1];
      }
    }

    const outDir = path.join(__dirname, "output", `ep${episodeNum}`);
    fs.mkdirSync(outDir, { recursive: true });

    if (cutPlan) {
      // Save structured JSON
      const cutsPath = path.join(outDir, "drama_cuts.json");
      fs.writeFileSync(cutsPath, JSON.stringify(cutPlan, null, 2), "utf-8");
      log("SYSTEM", `Structured cut plan saved: ${cutsPath}`);

      const numCuts = cutPlan.cuts ? cutPlan.cuts.length : 0;
      const numIssues = cutPlan.validation_issues ? cutPlan.validation_issues.length : 0;
      log("SYSTEM", `${numCuts} cuts, ${numIssues} validation issues.`);

      // Print validation issues
      if (cutPlan.validation_issues && cutPlan.validation_issues.length > 0) {
        console.log("\n  VALIDATION ISSUES:");
        for (const issue of cutPlan.validation_issues) {
          console.log(`    - [${issue.agent}] ${issue.issue}`);
          if (issue.correction) console.log(`      Fix: ${issue.correction}`);
        }
      }

      // Print cuts summary
      if (cutPlan.cuts) {
        console.log("\n  CUTS:");
        for (const cut of cutPlan.cuts) {
          console.log(`    Cut ${cut.cut_number} (${cut.scene_id}): ${cut.description?.substring(0, 80)}...`);
        }
      }
    } else {
      // Save raw text as fallback
      const rawPath = path.join(outDir, "drama_cuts_raw.txt");
      fs.writeFileSync(rawPath, arbiterResult.output, "utf-8");
      log("SYSTEM", `Raw output saved: ${rawPath}`);
      console.log(arbiterResult.output);
    }

    // Also save the script for the Python pipeline to use
    const scriptPath = path.join(outDir, "script.txt");
    fs.writeFileSync(scriptPath, script, "utf-8");
    log("SYSTEM", `Script saved: ${scriptPath}`);

  } else {
    console.log(`[Final Arbiter failed: ${arbiterResult.error}]`);
    console.log("");
    console.log("Individual agent outputs (for reference):");
    for (const r of results) {
      if (r.success) {
        console.log(`\n--- ${r.name} ---\n${r.output}`);
      }
    }
  }

  console.log("");
  console.log("=".repeat(70));
  log("SYSTEM", "Pipeline complete.");
  console.log("=".repeat(70));
}

// ---------------------------------------------------------------------------
// ENTRY POINT
// ---------------------------------------------------------------------------

main().catch((err) => {
  log("SYSTEM", `Unexpected error: ${err.message}`);
  process.exit(1);
});

// ---------------------------------------------------------------------------
// HOW TO RUN:
// 1. Make sure Ollama is running: ollama serve
// 2. Make sure your model is pulled: ollama pull gemma4:26b
// 3. Ensure .env has GEMINI_API_KEY=your-key-here
// 4. Run: node drama_pipeline.js
// 5. Paste your script, then type ---END--- on its own line and press Enter
// ---------------------------------------------------------------------------
