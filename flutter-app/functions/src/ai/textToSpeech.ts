/**
 * textToSpeech — HTTPS Callable
 * Takes text and uses Google Cloud Text-to-Speech
 * to generate speech audio, returned as base64.
 */
import { onCall, HttpsError } from "firebase-functions/v2/https";
import textToSpeechLib from "@google-cloud/text-to-speech";

const LOCATION = "asia-southeast1";

// Language code to voice name mapping for natural-sounding voices
const VOICE_MAP: Record<string, { name: string; languageCode: string }> = {
    "en": { name: "en-US-Neural2-J", languageCode: "en-US" },
    "en-US": { name: "en-US-Neural2-J", languageCode: "en-US" },
    "en-GB": { name: "en-GB-Neural2-B", languageCode: "en-GB" },
    "ms": { name: "ms-MY-Standard-A", languageCode: "ms-MY" },
    "zh": { name: "cmn-CN-Standard-A", languageCode: "cmn-CN" },
    "zh-CN": { name: "cmn-CN-Standard-A", languageCode: "cmn-CN" },
    "ja": { name: "ja-JP-Neural2-B", languageCode: "ja-JP" },
    "ko": { name: "ko-KR-Neural2-A", languageCode: "ko-KR" },
    "hi": { name: "hi-IN-Neural2-A", languageCode: "hi-IN" },
    "ta": { name: "ta-IN-Standard-A", languageCode: "ta-IN" },
    "th": { name: "th-TH-Standard-A", languageCode: "th-TH" },
    "vi": { name: "vi-VN-Neural2-A", languageCode: "vi-VN" },
    "id": { name: "id-ID-Standard-A", languageCode: "id-ID" },
    "fr": { name: "fr-FR-Neural2-A", languageCode: "fr-FR" },
};

export const textToSpeech = onCall(
    { region: LOCATION, memory: "512MiB", timeoutSeconds: 60 },
    async (request) => {
        if (!request.auth) {
            throw new HttpsError("unauthenticated", "User must be authenticated.");
        }

        const { text, lang } = request.data as {
            text: string;
            lang?: string;
        };

        if (!text || typeof text !== "string" || text.trim() === "") {
            throw new HttpsError(
                "invalid-argument",
                "text must be a non-empty string."
            );
        }

        const language = lang || "en";

        try {
            const client = new textToSpeechLib.TextToSpeechClient();

            // Get voice config for language, fallback to en-US
            const voiceConfig = VOICE_MAP[language] || VOICE_MAP["en"];

            const [response] = await client.synthesizeSpeech({
                input: { text: text },
                voice: {
                    languageCode: voiceConfig.languageCode,
                    name: voiceConfig.name,
                },
                audioConfig: {
                    audioEncoding: "MP3",
                    speakingRate: 1.0,
                    pitch: 0.0,
                },
            });

            const audioBase64 = Buffer.from(
                response.audioContent as Uint8Array
            ).toString("base64");

            console.log(
                `textToSpeech [${language}]: "${text.substring(0, 50)}..." → ${audioBase64.length} bytes`
            );

            return { audioBase64, contentType: "audio/mp3" };
        } catch (error: any) {
            console.error("textToSpeech error:", error);
            throw new HttpsError("internal", `TTS API error: ${error.message}`);
        }
    }
);
