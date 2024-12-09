import { BedrockRuntimeClient, InvokeModelWithResponseStreamCommand } from "@aws-sdk/client-bedrock-runtime"
import { Anthropic } from "@anthropic-ai/sdk"
import { ApiHandler } from "../"
import { ApiHandlerOptions, BedrockModelId, ModelInfo, bedrockDefaultModelId, bedrockModels } from "../../shared/api"
import { ApiStream } from "../transform/stream"

export class AwsBedrockRuntimeHandler implements ApiHandler {
    private options: ApiHandlerOptions
    private client: BedrockRuntimeClient

    constructor(options: ApiHandlerOptions) {
        this.options = options
        this.client = new BedrockRuntimeClient({
            credentials: options.awsAccessKey ? {
                accessKeyId: options.awsAccessKey,
                secretAccessKey: options.awsSecretKey || "",
                sessionToken: options.awsSessionToken,
            } : undefined,
            region: options.awsRegion || "us-east-1",
        })
    }

    async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
        const modelConfig = this.getModel()
        let modelId = modelConfig.id
        if (this.options.awsUseCrossRegionInference) {
            const regionPrefix = (this.options.awsRegion || "").slice(0, 3)
            modelId = regionPrefix === "us-" ? `us.${modelId}` :
                     regionPrefix === "eu-" ? `eu.${modelId}` : modelId
        }

        // Convert messages to the format expected by Bedrock Runtime
        const prompt = {
            anthropic_version: "bedrock-2024-02-20",
            messages: messages.map(msg => ({
                role: msg.role,
                content: typeof msg.content === 'string' ? msg.content : this.formatContent(msg.content)
            })),
            system: systemPrompt,
            max_tokens: modelConfig.info.maxTokens || 8192,
            temperature: 0
        }

        // Add nova-specific configurations if using a Nova model
        if (this.isNovaModel()) {
            Object.assign(prompt, {
                cache_control: modelConfig.info.supportsPromptCache ? {
                    enable_prompt_caching: true,
                    prompt_routing: {
                        enable_routing: true,
                        temperature: 0
                    }
                } : undefined,
                nova_config: {
                    optimized_response: true,
                    context_adaptation: true,
                }
            })
        }

        const command = new InvokeModelWithResponseStreamCommand({
            modelId,
            contentType: "application/json",
            accept: "application/json",
            body: JSON.stringify(prompt),
        })

        try {
            const response = await this.client.send(command)
            if (!response.body) {
                throw new Error('No response stream available')
            }

            let hasStarted = false
            let inputTokens = 0
            let outputTokens = 0

            for await (const chunk of response.body) {
                if (chunk.chunk?.bytes) {
                    const data = JSON.parse(new TextDecoder().decode(chunk.chunk.bytes))
                    
                    switch (data.type) {
                        case "message_start":
                            if (data.message?.usage) {
                                inputTokens = data.message.usage.input_tokens || 0
                                outputTokens = data.message.usage.output_tokens || 0
                                yield {
                                    type: "usage",
                                    inputTokens,
                                    outputTokens
                                }
                            }
                            break

                        case "content_block_delta":
                            if (data.delta?.text) {
                                if (!hasStarted) {
                                    hasStarted = true
                                }
                                yield {
                                    type: "text",
                                    text: data.delta.text
                                }
                            }
                            break

                        case "message_delta":
                            if (data.usage?.output_tokens) {
                                outputTokens = data.usage.output_tokens
                                yield {
                                    type: "usage",
                                    inputTokens,
                                    outputTokens
                                }
                            }
                            break

                        case "message_stop":
                            yield {
                                type: "usage",
                                inputTokens,
                                outputTokens
                            }
                            break
                    }
                }
            }

            if (!hasStarted) {
                throw new Error('No content was generated by the model')
            }

        } catch (error: any) {
            console.error("Error in Bedrock Runtime stream:", error)
            yield {
                type: "text",
                text: `Error: ${error.message}`
            }
            yield {
                type: "usage",
                inputTokens: 0,
                outputTokens: 0
            }
            throw error
        }
    }

    private formatContent(content: any[]): string {
        return content.map(block => {
            switch (block.type) {
                case 'text':
                    return block.text
                case 'image':
                    return `[Image: ${block.source}]`
                case 'tool_use':
                    return `[Tool: ${block.name}]`
                default:
                    return '[Unknown Block]'
            }
        }).join(' ')
    }

    getModel(): { id: BedrockModelId; info: ModelInfo } {
        const modelId = this.options.apiModelId
        if (modelId && modelId in bedrockModels) {
            const id = modelId as BedrockModelId
            return { id, info: bedrockModels[id] }
        }
        return { 
            id: bedrockDefaultModelId, 
            info: bedrockModels[bedrockDefaultModelId] 
        }
    }

    /**
     * Check if the current model is a Nova model
     */
    private isNovaModel(): boolean {
        return this.getModel().id.toLowerCase().includes('nova')
    }

    /**
     * Check if we should use the Bedrock Runtime API
     * This is true for Nova models or when explicitly requested
     */
    static shouldUseRuntime(options: ApiHandlerOptions): boolean {
        const modelId = options.apiModelId?.toLowerCase() || bedrockDefaultModelId
        return modelId.includes('nova') || options.useBedrockRuntime === true
    }
}
