package main

import (
	"encoding/json"
	"fmt"
	"os"
	"text/template"
)

func main() {
	// 1. Read JSON - Ensure the path is correct relative to where you run 'go run'
	// If you run from the root, use "data/holdings.json".
	// If you run from inside /sectors, use "../data/holdings.json".
	jsonData, err := os.ReadFile("data/holdings.json")
	if err != nil {
		fmt.Printf("❌ Error reading JSON: %v\n", err)
		os.Exit(1)
	}

	var sectors map[string][]string
	if err := json.Unmarshal(jsonData, &sectors); err != nil {
		fmt.Printf("❌ Error parsing JSON: %v\n", err)
		os.Exit(1)
	}

	// Debug: Print count to Action log to verify data is loaded
	fmt.Printf("✅ Loaded %d sectors from JSON\n", len(sectors))

	// 2. Setup Template
	// Note: Use the filename that matches your .tmpl file exactly
	tmpl, err := template.New("sectors.tmpl").Funcs(template.FuncMap{
		"len": func(arr []string) int { return len(arr) },
	}).ParseFiles("templates/sectors.tmpl")

	if err != nil {
		fmt.Printf("❌ Error loading template: %v\n", err)
		os.Exit(1)
	}

	// 3. Create Output
	f, err := os.Create("SectorLib.pine")
	if err != nil {
		fmt.Printf("❌ Error creating file: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	// 4. Execute - Pass the map under the key "Sectors" to match your .tmpl
	err = tmpl.Execute(f, map[string]interface{}{
		"Sectors": sectors,
	})
	if err != nil {
		fmt.Printf("❌ Error executing template: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("🚀 SectorLib.pine generated successfully!")
}
