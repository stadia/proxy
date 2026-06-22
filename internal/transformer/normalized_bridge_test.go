package transformer

import (
	"encoding/json"
	"testing"

	"github.com/routatic/proxy/internal/config"
	"github.com/routatic/proxy/internal/core"
)

func TestNormalizedToAnthropic_SystemPromptWithNewline(t *testing.T) {
	req := &core.NormalizedRequest{
		Model:        "minimax-m3",
		SystemPrompt: "Line one\nLine two\nLine three",
		MaxTokens:    100,
		Messages: []core.NormalizedMessage{
			{Role: "user", Content: "Hello"},
		},
	}

	anthropicReq := NormalizedToAnthropic(req, config.ModelConfig{ModelID: "minimax-m3"})

	// The bug: marshaling the request failed when the system prompt contained
	// unescaped newlines because we built the RawMessage by wrapping the raw
	// string in quotes instead of JSON-encoding it.
	_, err := json.Marshal(anthropicReq)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	if got := anthropicReq.SystemText(); got != req.SystemPrompt {
		t.Fatalf("system text mismatch: got %q, want %q", got, req.SystemPrompt)
	}
}

func TestNormalizedToAnthropic_MessageContentWithNewline(t *testing.T) {
	req := &core.NormalizedRequest{
		Model:     "minimax-m3",
		MaxTokens: 100,
		Messages: []core.NormalizedMessage{
			{Role: "user", Content: "Hello\nWorld"},
		},
	}

	anthropicReq := NormalizedToAnthropic(req, config.ModelConfig{ModelID: "minimax-m3"})

	_, err := json.Marshal(anthropicReq)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	blocks := anthropicReq.Messages[0].ContentBlocks()
	if len(blocks) != 1 || blocks[0].Text != "Hello\nWorld" {
		t.Fatalf("unexpected content blocks: %+v", blocks)
	}
}

func TestNormalizedToResponses_SystemPromptWithNewline(t *testing.T) {
	req := &core.NormalizedRequest{
		Model:        "gpt-5",
		SystemPrompt: "Line one\nLine two",
		MaxTokens:    100,
		Messages: []core.NormalizedMessage{
			{Role: "user", Content: "Hello\nWorld"},
		},
	}

	responsesReq := NormalizedToResponses(req, config.ModelConfig{ModelID: "gpt-5"})

	_, err := json.Marshal(responsesReq)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}
}
