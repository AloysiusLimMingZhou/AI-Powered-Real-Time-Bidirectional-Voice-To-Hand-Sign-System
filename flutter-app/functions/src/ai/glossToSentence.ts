/**
 * glossToSentence — HTTPS Callable
 * Takes raw gloss words (e.g., ["HELLO", "HOW", "YOU"]) and uses Gemini
 * to convert them into a natural English sentence.
 */
import { onCall, HttpsError } from "firebase-functions/v2/https";
import { VertexAI } from "@google-cloud/vertexai";

const PROJECT_ID = "ai-real-time-voice-to-sign";
const LOCATION = "asia-southeast1";
const MODEL = "gemini-2.5-flash";

export const glossToSentence = onCall(
    { region: LOCATION, memory: "512MiB", timeoutSeconds: 60 },
    async (request) => {
        // Ensure user is authenticated
        if (!request.auth) {
            throw new HttpsError("unauthenticated", "User must be authenticated.");
        }

        const { glossWords, lang } = request.data as {
            glossWords: string[];
            lang?: string;
        };

        if (!glossWords || !Array.isArray(glossWords) || glossWords.length === 0) {
            throw new HttpsError(
                "invalid-argument",
                "glossWords must be a non-empty array of strings."
            );
        }

        const language = lang || "en";

        try {
            const vertexAI = new VertexAI({ project: PROJECT_ID, location: LOCATION });
            const model = vertexAI.getGenerativeModel({ model: MODEL });

            const prompt = `You are a sign language interpreter assistant. Convert the following sign language gloss sequence into a natural, grammatically correct ${language} sentence. 

Gloss words: ${glossWords.join(" ")}

Rules:
- Output ONLY the natural sentence, nothing else.
- Do not add quotes or formatting.
- Make the sentence sound natural and conversational.
- If the gloss seems incomplete, make your best guess at the intended meaning.`;

            const result = await model.generateContent(prompt);
            const response = result.response;
            const sentence =
                response.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || "";

            console.log(
                `glossToSentence: [${glossWords.join(", ")}] → "${sentence}"`
            );

            return { sentence };
        } catch (error: any) {
            console.error("glossToSentence error:", error);
            throw new HttpsError("internal", `Gemini API error: ${error.message}`);
        }
    }
);
