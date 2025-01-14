package azurrr

import (
	"context"
	"fmt"
	"github.com/Azure/azure-sdk-for-go/sdk/ai/azopenai"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/to"
	"log"
	"os"
)

var endpointType azopenai.OnYourDataVectorizationSourceType = "endpoint"
var authType azopenai.OnYourDataVectorSearchAuthenticationType = "api_key"

func StartAzure(ctx context.Context) {
	azureOpenAIKey := os.Getenv("AZURE_OPENAI_API_KEY")
	modelDeploymentID := os.Getenv("DEPLOYMENT_NAME")
	azureOpenAIEndpoint := os.Getenv("AOAI_ENDPOINT_URL")
	searchIndex := os.Getenv("SEARCH_INDEX_NAME")
	searchEndpoint := os.Getenv("SEARCH_ENDPOINT")
	searchAPIKey := os.Getenv("SEARCH_KEY")
	embedingEndpoint := os.Getenv("EMBEDDING_ENDPOINT")

	log.Printf("Azure OpenAI Endpoint: %s", azureOpenAIEndpoint)
	log.Printf("Model Deployment ID: %s", modelDeploymentID)
	log.Printf("Search Endpoint: %s", searchEndpoint)
	log.Printf("Search Index: %s", searchIndex)
	keyCredential := azcore.NewKeyCredential(azureOpenAIKey)

	client, err := azopenai.NewClientWithKeyCredential(azureOpenAIEndpoint, keyCredential, nil)
	if err != nil {
		log.Fatalf("ERROrwerweR: %+v", err)
	}

	systemPrompt := "You are an AI assistant that helps people find information "
	userMessage := "tell me a joke"
	queryType := azopenai.AzureSearchQueryType("vector_simple_hybrid")
	messages := []azopenai.ChatRequestMessageClassification{
		&azopenai.ChatRequestSystemMessage{Content: azopenai.NewChatRequestSystemMessageContent(systemPrompt)},
		&azopenai.ChatRequestUserMessage{Content: azopenai.NewChatRequestUserMessageContent(userMessage)},
	}

	resp, err := client.GetChatCompletions(ctx, azopenai.ChatCompletionsOptions{
		Messages:         messages,
		MaxTokens:        to.Ptr[int32](800),
		Temperature:      to.Ptr[float32](0.7),
		TopP:             to.Ptr[float32](0.95),
		FrequencyPenalty: to.Ptr[float32](0),
		PresencePenalty:  to.Ptr[float32](0),

		AzureExtensionsOptions: []azopenai.AzureChatExtensionConfigurationClassification{
			&azopenai.AzureSearchChatExtensionConfiguration{
				Parameters: &azopenai.AzureSearchChatExtensionParameters{
					Endpoint:  &searchEndpoint,
					IndexName: &searchIndex,
					Authentication: &azopenai.OnYourDataAPIKeyAuthenticationOptions{
						Key: &searchAPIKey,
					},
					Strictness:    to.Ptr[int32](5),
					InScope:       to.Ptr[bool](true),
					TopNDocuments: to.Ptr[int32](5),
					QueryType:     &queryType,
					EmbeddingDependency: &azopenai.OnYourDataEndpointVectorizationSource{
						Authentication: &azopenai.OnYourDataVectorSearchAPIKeyAuthenticationOptions{
							Type: &authType,
							Key:  &azureOpenAIKey,
						},
						Endpoint: &embedingEndpoint,
						Type:     &endpointType,
					},
					SemanticConfiguration: to.Ptr("azureml-default"),
				},
			},
		},
		DeploymentName: &modelDeploymentID,
	}, nil)

	if err != nil {
		log.Fatalf("ERROR: %+v", err)
	}

	fmt.Fprintf(os.Stderr, "Extensions Context Role: %s\nExtensions Context (length): %d\n", *resp.Choices[0].Message.Role, len(*resp.Choices[0].Message.Content))
	fmt.Fprintf(os.Stderr, "ChatRole: %s\nChat content: %s\n", *resp.Choices[0].Message.Role, *resp.Choices[0].Message.Content)
}
