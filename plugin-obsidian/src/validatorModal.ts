import { App, Modal, Setting, ButtonComponent, MarkdownRenderer } from 'obsidian';
import { ADRValidationResponse, SecurityRisk, Contradiction, RelatedADR } from './api';
import ADRValidatorPlugin from '../main';

export class ValidatorModal extends Modal {
    constructor(app: App, private plugin: ADRValidatorPlugin, private result: ADRValidationResponse) {
        super(app);
    }

    onOpen() {
        const { contentEl } = this;
        contentEl.empty();
        contentEl.addClass('adr-validator-modal');

        this.renderHeader();
        this.renderStatus();
        this.renderContradictions();
        this.renderSecurityRisks();
        this.renderRecommendations();
        this.renderRelatedADRs();
        this.renderTechnologies();
        this.renderFooter();
    }

    private renderHeader() {
        const { contentEl } = this;
        contentEl.createEl('h2', {
            text: '🔍 ADR Validation Results',
            cls: 'adr-modal-title'
        });
    }

    private renderStatus() {
        const { contentEl } = this;
        const statusEl = contentEl.createDiv('adr-status-container');

        const statusIcons: Record<string, string> = {
            'approved': '✅',
            'needs_review': '⚠️',
            'needs_minor_revision': '🟡',
            'rejected': '❌'
        };

        const statusColors: Record<string, string> = {
            'approved': '#22c55e',
            'needs_review': '#ef4444',
            'needs_minor_revision': '#eab308',
            'rejected': '#dc2626'
        };

        const icon = statusIcons[this.result.status] || '❓';
        const color = statusColors[this.result.status] || '#6b7280';

        statusEl.createEl('div', {
            text: `${icon} ${this.result.status.replace('_', ' ').toUpperCase()}`,
            cls: 'adr-status-badge'
        }).style.setProperty('background-color', color);

        statusEl.createEl('p', {
            text: this.result.message,
            cls: 'adr-status-message'
        });
    }

    private renderContradictions() {
        const { contentEl } = this;
        const contradictions = this.result.contradictions || [];

        if (contradictions.length === 0) return;

        contentEl.createEl('h3', {
            text: '⚠️ Contradictions Detected',
            cls: 'adr-section-title adr-section-warning'
        });

        contradictions.forEach((c: Contradiction) => {
            const div = contentEl.createDiv('adr-contradiction');
            div.createEl('strong', { text: `[${c.severity.toUpperCase()}] ${c.related_adr_title}` });
            div.createEl('p', { text: c.description });
            if (c.source) {
                div.createEl('small', { text: `Source: ${c.source}` });
            }
        });
    }

    private renderSecurityRisks() {
        const { contentEl } = this;
        const risks = this.result.security_risks || [];

        if (risks.length === 0) return;

        contentEl.createEl('h3', {
            text: '🛡️ Security Risks',
            cls: 'adr-section-title adr-section-danger'
        });

        risks.forEach((r: SecurityRisk) => {
            const div = contentEl.createDiv('adr-risk');
            div.createEl('strong', { text: `[${r.severity.toUpperCase()}] ${r.type.replace('_', ' ')}` });
            div.createEl('p', { text: r.description });

            if (r.secure_alternative) {
                const codeBlock = div.createEl('pre', { cls: 'adr-code-secure' });
                codeBlock.createEl('code', { text: r.secure_alternative });
            }
        });
    }

    private renderRecommendations() {
        const { contentEl } = this;
        const recommendations = this.result.recommendations || [];

        if (recommendations.length === 0) return;

        contentEl.createEl('h3', {
            text: '💡 Recommendations & AI Analysis',
            cls: 'adr-section-title adr-section-info'
        });

        const container = contentEl.createDiv('adr-recommendations-container');
        
        recommendations.forEach(rec => {
            const item = container.createDiv('adr-recommendation-item');
            
            if (rec.includes('AI Analysis:')) {
                item.addClass('adr-ai-critique');
                const content = rec.replace('AI Analysis:', '').trim();
                MarkdownRenderer.renderMarkdown(content, item, '', this.plugin);
            } else {
                MarkdownRenderer.renderMarkdown(rec, item, '', this.plugin);
            }
        });
    }

    private renderRelatedADRs() {
        const { contentEl } = this;
        const related = this.result.related_adrs || [];

        if (related.length === 0) return;

        contentEl.createEl('h3', {
            text: '📚 Related ADRs',
            cls: 'adr-section-title'
        });

        const ul = contentEl.createEl('ul', 'adr-related');
        related.forEach((adr: RelatedADR) => {
            const li = ul.createEl('li');
            li.createEl('strong', { text: adr.title });
            li.createEl('span', {
                text: ` (${(adr.similarity * 100).toFixed(1)}% match - ${adr.status})`
            });
        });
    }

    private renderTechnologies() {
        const { contentEl } = this;
        const techs = this.result.detected_technologies || [];

        if (techs.length === 0) return;

        contentEl.createEl('h3', {
            text: '🔧 Detected Technologies',
            cls: 'adr-section-title'
        });

        const tags = contentEl.createDiv('adr-tech-tags');
        techs.forEach(tech => {
            tags.createEl('span', {
                text: tech,
                cls: 'adr-tech-tag'
            });
        });
    }

    private renderFooter() {
        const { contentEl } = this;

        new Setting(contentEl)
            .addButton((btn: ButtonComponent) => {
                btn.setButtonText('Close')
                    .setCta()
                    .onClick(() => this.close());
            });
    }

    onClose() {
        const { contentEl } = this;
        contentEl.empty();
    }
}
