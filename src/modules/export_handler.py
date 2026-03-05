import pandas as pd
from datetime import datetime
import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors

class ExportHandler:
    """
    Export analysis results to CSV and PDF formats.
    """

    def __init__(self, output_folder: str = 'output'):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

    def export_to_csv(self, df: pd.DataFrame, filename: str) -> str:
        """
        Export DataFrame to CSV.

        Args:
            df: DataFrame to export
            filename: Output filename (without extension)

        Returns:
            Path to saved file
        """
        filepath = os.path.join(self.output_folder, f"{filename}.csv")

        df.to_csv(filepath, index=False, encoding='utf-8')

        return filepath

    def export_messages_with_analysis(self, df: pd.DataFrame, filename: str = "messages_analysis") -> str:
        """
        Export messages with all analysis columns to CSV.

        Args:
            df: Analyzed DataFrame
            filename: Output filename

        Returns:
            Path to saved file
        """
        export_columns = [
            'datetime', 'user', 'message',
            'sentiment_vader', 'sentiment_compound',
            'emotion', 'emotion_score',
            'is_toxic', 'toxicity_score'
        ]

        available_columns = [col for col in export_columns if col in df.columns]

        df_export = df[available_columns].copy()

        return self.export_to_csv(df_export, filename)

    def export_user_statistics(self, stats: dict, filename: str = "user_statistics") -> str:
        """
        Export user statistics to CSV.

        Args:
            stats: Dictionary of user statistics
            filename: Output filename

        Returns:
            Path to saved file
        """
        df_stats = pd.DataFrame.from_dict(stats, orient='index')

        return self.export_to_csv(df_stats, filename)

    def create_pdf_report(self, df: pd.DataFrame, analytics_summary: dict,
                         sentiment_dist: dict, emotion_dist: dict,
                         behavioral_stats: dict, filename: str = "sentiment_analysis_report") -> str:
        """
        Create comprehensive PDF report.

        Args:
            df: Analyzed DataFrame
            analytics_summary: Summary statistics from analytics
            sentiment_dist: Sentiment distribution
            emotion_dist: Emotion distribution
            behavioral_stats: Behavioral analysis stats
            filename: Output filename

        Returns:
            Path to saved file
        """
        filepath = os.path.join(self.output_folder, f"{filename}.pdf")

        doc = SimpleDocTemplate(filepath, pagesize=letter)
        elements = []

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#00A699'),
            spaceAfter=30,
            alignment=1
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#00A699'),
            spaceAfter=12,
            spaceBefore=12
        )

        elements.append(Paragraph("WhatsApp Sentiment Analysis Report", title_style))
        elements.append(Spacer(1, 0.3*inch))

        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))

        elements.append(Paragraph("Overview", heading_style))

        overview_data = [
            ['Metric', 'Value'],
            ['Total Messages', str(analytics_summary.get('total_messages', 'N/A'))],
            ['Unique Users', str(analytics_summary.get('unique_users', 'N/A'))],
            ['Date Range', f"{analytics_summary.get('date_range_start', 'N/A')} to {analytics_summary.get('date_range_end', 'N/A')}"],
            ['Avg Messages/Day', f"{analytics_summary.get('avg_messages_per_day', 0):.2f}"],
            ['Most Active User', str(analytics_summary.get('most_active_user', 'N/A'))],
        ]

        overview_table = Table(overview_data, colWidths=[3*inch, 3*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00A699')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        elements.append(overview_table)
        elements.append(Spacer(1, 0.3*inch))

        elements.append(Paragraph("Sentiment Analysis", heading_style))

        sentiment_data = [['Sentiment', 'Count', 'Percentage']]

        for sentiment, stats in sentiment_dist.items():
            sentiment_data.append([
                sentiment,
                str(stats.get('count', 0)),
                f"{stats.get('percentage', 0):.2f}%"
            ])

        sentiment_table = Table(sentiment_data, colWidths=[2*inch, 2*inch, 2*inch])
        sentiment_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E85D75')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        elements.append(sentiment_table)
        elements.append(Spacer(1, 0.3*inch))

        elements.append(Paragraph("Emotion Analysis", heading_style))

        emotion_data = [['Emotion', 'Count', 'Percentage']]

        for emotion, stats in emotion_dist.items():
            emotion_data.append([
                emotion,
                str(stats.get('count', 0)),
                f"{stats.get('percentage', 0):.2f}%"
            ])

        emotion_table = Table(emotion_data, colWidths=[2*inch, 2*inch, 2*inch])
        emotion_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4ECDC4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        elements.append(emotion_table)
        elements.append(Spacer(1, 0.3*inch))

        if behavioral_stats:
            elements.append(Paragraph("Conversation Health", heading_style))

            health_data = [
                ['Metric', 'Value'],
                ['Health Score', f"{behavioral_stats.get('health_score', 0):.3f}"],
                ['Status', behavioral_stats.get('status', 'N/A')],
                ['Avg Sentiment', f"{behavioral_stats.get('avg_sentiment', 0):.3f}"],
                ['Toxic Messages %', f"{behavioral_stats.get('toxic_percentage', 0):.2f}%"],
            ]

            health_table = Table(health_data, colWidths=[3*inch, 3*inch])
            health_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#95E1D3')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightyellow),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            elements.append(health_table)

        doc.build(elements)

        return filepath

    def get_export_summary(self) -> dict:
        """Get summary of exported files"""
        files = os.listdir(self.output_folder) if os.path.exists(self.output_folder) else []

        return {
            'output_folder': self.output_folder,
            'exported_files': files,
            'total_files': len(files)
        }
