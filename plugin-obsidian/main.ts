import { Plugin, TFile, Notice, MarkdownView, Setting } from 'obsidian';
import { ADRValidatorSettingTab, DEFAULT_SETTINGS, ADRValidatorSettings } from './src/settingsTab';
import { ValidatorModal } from './src/validatorModal';
import { ADRAPIClient, ADRValidationResponse } from './src/api';

export default class ADRValidatorPlugin extends Plugin {
    settings: ADRValidatorSettings;
    apiClient: ADRAPIClient;
    private debounceTimer: NodeJS.Timeout | null = null;

    async onload() {
        await this.loadSettings();

        this.apiClient = new ADRAPIClient(
            this.settings.apiEndpoint,
            this.settings.apiKey
        );

        this.addRibbonIcon('shield-check', 'ADR Validator', () => {
            this.validateCurrentADR();
        });

        this.addCommand({
            id: 'validate-adr',
            name: 'Validate ADR',
            callback: () => this.validateCurrentADR()
        });

        this.addCommand({
            id: 'index-adr',
            name: 'Index ADR in Qdrant',
            callback: () => this.indexCurrentADR()
        });

        this.addCommand({
            id: 'check-api-health',
            name: 'Check API Health',
            callback: () => this.checkAPIHealth()
        });

        this.addSettingTab(new ADRValidatorSettingTab(this.app, this));

        if (this.settings.realtimeValidation) {
            this.registerEvent(
                this.app.workspace.on('editor-change', (editor) => {
                    this.debouncedValidate(editor.getValue());
                })
            );
        }

        console.log('ADR Security Validator loaded');
    }

    onunload() {
        console.log('ADR Security Validator unloaded');
    }

    async loadSettings() {
        this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
    }

    async saveSettings() {
        // Sanitize endpoint URL (remove trailing slash)
        if (this.settings.apiEndpoint.endsWith('/')) {
            this.settings.apiEndpoint = this.settings.apiEndpoint.slice(0, -1);
        }
        
        await this.saveData(this.settings);
        
        // Update API client with new settings
        this.apiClient = new ADRAPIClient(
            this.settings.apiEndpoint,
            this.settings.apiKey
        );
    }

    private debouncedValidate = (content: string) => {
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }
        this.debounceTimer = setTimeout(() => {
            this.validateADR(content);
        }, 2000);
    };

    async validateCurrentADR() {
        const activeView = this.app.workspace.getActiveViewOfType(MarkdownView);
        if (!activeView) {
            new Notice('No active file');
            return;
        }

        const content = activeView.editor.getValue();
        await this.validateADR(content);
    }

    async validateADR(content: string) {
        if (!content || content.trim().length < 10) {
            return;
        }

        const titleMatch = content.match(/^#\s+(.+)$/m);
        const title = titleMatch ? titleMatch[1] : 'Untitled ADR';

        // Show persistent loading notice
        const loadingNotice = new Notice('🧠 AI Architect is analyzing your ADR...\nEstimated time: 5-8 seconds', 0);
        
        try {
            const startTime = Date.now();
            
            const result = await this.apiClient.validateADR({
                title,
                content,
                context: this.settings.additionalContext || undefined
            });

            const duration = ((Date.now() - startTime) / 1000).toFixed(1);
            loadingNotice.hide(); // Remove loading notice

            new ValidatorModal(this.app, this, result).open();

            const riskCount = result.security_risks?.length || 0;
            const contradictionCount = result.contradictions?.length || 0;

            if (riskCount > 0 || contradictionCount > 0) {
                new Notice(
                    `✅ Analysis complete in ${duration}s\n⚠️ ${contradictionCount} contradictions, ${riskCount} security risks`,
                    6000
                );
            } else {
                new Notice(`✅ ADR analyzed in ${duration}s. No major issues found.`, 4000);
            }

        } catch (error) {
            loadingNotice.hide();
            console.error('Validation error:', error);
            new Notice('❌ Validation failed. Check API connection and AMD Cloud status.', 5000);
        }
    }

    async indexCurrentADR() {
        const activeFile = this.app.workspace.getActiveFile();
        if (!activeFile) {
            new Notice('No active file');
            return;
        }

        const content = await this.app.vault.read(activeFile);
        const title = activeFile.basename;

        const category = this.detectCategory(content);

        try {
            await this.apiClient.indexADR({
                title,
                content,
                metadata: {
                    path: activeFile.path,
                    category,
                    status: 'proposed'
                }
            });

            new Notice('✅ ADR indexed successfully', 3000);
        } catch (error) {
            console.error('Indexing error:', error);
            new Notice('❌ Failed to index ADR', 5000);
        }
    }

    async checkAPIHealth() {
        try {
            const healthy = await this.apiClient.healthCheck();
            if (healthy) {
                new Notice('✅ API is healthy', 2000);
            } else {
                new Notice('⚠️ API returned unhealthy status', 3000);
            }
        } catch (error) {
            new Notice('❌ Cannot reach API', 3000);
        }
    }

    private detectCategory(content: string): string {
        const lower = content.toLowerCase();
        if (lower.includes('database') || lower.includes('sql') || lower.includes('mongodb')) return 'database';
        if (lower.includes('security') || lower.includes('auth') || lower.includes('encryption')) return 'security';
        if (lower.includes('api') || lower.includes('http') || lower.includes('endpoint')) return 'api';
        if (lower.includes('container') || lower.includes('docker') || lower.includes('kubernetes')) return 'infrastructure';
        if (lower.includes('frontend') || lower.includes('ui') || lower.includes('react')) return 'frontend';
        return 'general';
    }
}