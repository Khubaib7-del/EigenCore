import type { Provider } from "../api";
import ai2 from "../icons/providers/ai2.svg?raw";
import deepseek from "../icons/providers/deepseek.svg?raw";
import gemma from "../icons/providers/gemma.svg?raw";
import huggingface from "../icons/providers/huggingface.svg?raw";
import meta from "../icons/providers/meta.svg?raw";
import microsoft from "../icons/providers/microsoft.svg?raw";
import mistral from "../icons/providers/mistral.svg?raw";
import qwen from "../icons/providers/qwen.svg?raw";
import tinyllama from "../icons/providers/tinyllama.svg?raw";

/** Monochrome provider marks (fill="currentColor"); sources in ui/CREDITS.md. */
export const PROVIDER_SVG: Record<Provider | "huggingface", string> = {
  tinyllama,
  microsoft,
  qwen,
  meta,
  mistral,
  gemma,
  deepseek,
  ai2,
  huggingface,
};

export const PROVIDER_NAME: Record<Provider | "huggingface", string> = {
  tinyllama: "TinyLlama (community)",
  microsoft: "Microsoft",
  qwen: "Qwen · Alibaba",
  meta: "Meta · Llama",
  mistral: "Mistral AI",
  gemma: "Google Gemma",
  deepseek: "DeepSeek",
  ai2: "Ai2 · OLMo",
  huggingface: "Hugging Face",
};

export function ProviderLogo({
  provider,
  size = 16,
  tile = false,
  title,
}: {
  provider: Provider | "huggingface";
  size?: number;
  /** Draw inside a neutral rounded tile. */
  tile?: boolean;
  title?: string;
}) {
  const svg = PROVIDER_SVG[provider].replace(/<title>[^<]*<\/title>/, "");
  const mark = (
    <span
      className="provider-mark"
      role="img"
      aria-label={title ?? PROVIDER_NAME[provider]}
      style={{ width: size, height: size }}
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
  if (!tile) return mark;
  return (
    <span className="provider-tile" style={{ width: size * 2.25, height: size * 2.25 }}>
      {mark}
    </span>
  );
}
